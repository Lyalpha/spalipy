import argparse
import itertools

from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy import linalg, interpolate
from scipy.ndimage import interpolation, map_coordinates
from scipy.spatial import distance

# SExtractor column definitions
X = 'X_IMAGE'
Y = 'Y_IMAGE'
FLUX = 'FLUX_BEST'
FWHM = 'FWHM_IMAGE'
FLAGS = 'FLAGS'

COLUMNS = [X, Y, FLUX, FWHM, FLAGS]


class Spalipy:
    """
    Detection-based astronomical image registration.

    Parameters
    ----------
    source_cat, template_cat : str or :class:`astropy.table.Table`
        The detection catalogue for the images. If `str` they should
        be the filenames of the SExtractor catalogues.
    source_fits : str or :class:`astropy.io.fits`
        The source image to be transformed.
    shape : None, str or :class:`astropy.io.fits.hdu.hdulist.HDUList`,
    optional
        The shape of the output image. If None, output shape is the
        same as `source_fits`. Otherwise pass a fits filename or
        class:`astropy.io.fits` instance and will take the shape from
        data in `hdu` in that fits file.
    hdu : int or str, optional
        The data in extension `hdu` of `source_fits` will be transformed.
        Also the hdu from which the shape is determined if `shape`
        specifies a fits file.
    ndets : int or float, optional
        The number of detections to use in the initial quad
        determination and detection matching. If 0 < `ndets` < 1 then
        will use this fraction of the shortest of `source_cat` and
        `template_cat` as `ndets`.
    nquaddets : integer
        The number of detections to make quads from.
    minquadsep : float
        Minimum distance in pixels between detections in a quad for it
        to be valid.
    minmatchdist : float
        Minimum matching distance between coordinates after the
        initial transformation to be considered a match.
    minnmatch : int
        Minimum number of matched dets for the initial transformation
        to be considered sucessful.
    spline_order : int
        The order in `x` and `y` of the spline surfaces used to
        correct the affine transformation.
    output_filename : None or str, optional
        The filename to write the transformed source file to. If None
        the file will not be written, the transformed data can be
        still accessed through Spalipy.source_data_transformed.
    overwrite : bool, optional
        Whether to overwrite `output_filename` if it exists.


    Example
    -------
    s = spalipy.Spalipy("source.cat", "template.cat", "source.fits")
    s.main()
    """

    def __init__(self, source_cat, template_cat, source_fits,
                 shape=None, hdu=0, ndets=0.5, nquaddets=20,
                 minquadsep=50, minmatchdist=5, minnmatch=200,
                 spline_order=3, output_filename=None, overwrite=True):

        if isinstance(source_cat, str):
            source_cat = Table.read(source_cat, format='ascii.sextractor')
        self.source_cat_full = source_cat

        if isinstance(template_cat, str):
            template_cat = Table.read(template_cat, format='ascii.sextractor')
        self.template_cat_full = template_cat

        if isinstance(source_fits, str):
            source_fits = fits.open(source_fits)
        self.source_fits = source_fits

        hdr = None
        if isinstance(shape, str):
            hdr = fits.getheader(shape, hdu)
        if isinstance(shape, fits.hdu.hdulist.HDUList):
            hdr = shape[hdu].header
        if shape is None:
            hdr = source_fits[hdu].header
        if hdr is not None:
            shape = (int(hdr['NAXIS1']), int(hdr['NAXIS2']))
        self.shape = shape

        self.hdu = hdu

        if isinstance(ndets, float):
            ntot = min(len(source_cat), len(template_cat))
            ndets = int(ndets * ntot)
        self.ndets = ndets

        if ndets < minnmatch:
            msg = ('ndet ({}) < minnmatch ({}) - will never find a suitable '
                   'transform'.format(ndets, minnmatch))
            raise ValueError(msg)

        self.nquaddets = nquaddets
        self.minquadsep = minquadsep
        self.minmatchdist = minmatchdist
        self.minnmatch = minnmatch
        self.spline_order = spline_order

        self.source_cat = self.trim_cat(source_cat)
        self.source_coo = get_det_coords(self.source_cat)
        self.template_cat = self.trim_cat(template_cat)
        self.template_coo = get_det_coords(self.template_cat)
        self.source_quadlist = []
        self.template_quadlist = []

        self.nmatch = 0
        self.source_matchdets = None
        self.template_matchdets = None
        self.affine_transform = None
        self.spline_transform = None

        self.output_filename = output_filename
        self.overwrite = overwrite

    def main(self):
        """
        Does everything
        """
        self.make_quadlist('source')
        self.make_quadlist('template')

        self.find_affine_transform()

        if self.affine_transform is None:
            print('{} matched dets is less than minimum required ({})'.format(
                self.nmatch, self.minnmatch))
            return

        if self.spline_order > 0:
            self.find_spline_transform()

        self.align()

    def make_quadlist(self, image, nquaddets=None, minquadsep=None):
        """
        Create a list of hashes for "quads" of the brightest sources
        in the detection catalogue.

        Parameters
        ----------
        image : str
            Should be "source" or "template" to indicate for which image to
            determine quadlist for.
        """
        if image == 'source':
            coo = self.source_coo
        elif image == 'template':
            coo = self.template_coo
        else:
            raise ValueError('image must be "source" or "template"')

        if nquaddets is None:
            nquaddets = self.nquaddets

        if minquadsep is None:
            minquadsep = self.minquadsep

        quadlist = []
        quad_idxs = itertools.combinations(range(nquaddets), 4)
        for quad_idx in quad_idxs:
            combo = coo[quad_idx, :]
            dists = distance.pdist(combo)
            if np.min(dists) > minquadsep:
                quadlist.append(quad(combo, dists))

        if image == 'source':
            self.source_quadlist = quadlist
        elif image == 'template':
            self.template_quadlist = quadlist

    def find_affine_transform(self, minmatchdist=None, minnmatch=None,
                              maxcands=10, minquaddist=0.005):
        """
        Use the quadlist hashes to determine an initial guess at an affine
        transformation and determine matched detections lists. Then refine
        the transformation using the matched detection lists.

        Parameters
        ----------
        maxcands : int, optional
            Max number of quadlist candidates to loop through to find initial
            transformation.
        minquaddist : float, optional
            Not really sure what this is, just copied from alipy.
        """
        if minmatchdist is None:
            minmatchdist = self.minmatchdist
        if minnmatch is None:
            minnmatch = self.minnmatch

        template_hash = np.array([q[1] for q in self.template_quadlist])
        source_hash = np.array([q[1] for q in self.source_quadlist])

        dists = distance.cdist(template_hash, source_hash)
        minddist_idx = np.argmin(dists, axis=0)
        mindist = np.min(dists, axis=0)
        best = np.argsort(mindist)

        # Use best initial guess at transformation to get list of matched dets
        for i in range(min(maxcands, len(best))):
            bi = best[i]
            template_quad = self.template_quadlist[minddist_idx[bi]]
            source_quad = self.source_quadlist[bi]
            # Get a quick (exact) transformation guess
            # using first two detections
            transform = calc_affine_transform(source_quad[0][:2],
                                              template_quad[0][:2])
            dist = mindist[bi]
            passed = False
            if dist < minquaddist:
                nmatch, source_matchdets, template_matchdets = \
                    self.match_dets(transform, minmatchdist=minmatchdist)
                if nmatch > minnmatch:
                    passed = True
                    break
        if passed:
            # Refine the transformation using the matched detections
            source_match_coo = get_det_coords(source_matchdets)
            template_match_coo = get_det_coords(template_matchdets)
            transform = calc_affine_transform(source_match_coo,
                                              template_match_coo)
            # Store the final matched detection tables and transform
            self.nmatch, self.source_matchdets, self.template_matchdets = \
                self.match_dets(transform, minmatchdist=minmatchdist)
            self.affine_transform = transform

    def find_spline_transform(self, spline_order=None):
        """
        Determine the residual `x` and `y` offsets between matched coordinates
        after affine transformation and fit 2D spline surfaces to describe the
        spatially-varying correction to be applied.
        """
        if spline_order is None:
            spline_order = self.spline_order

        # Get the source, after affine transformation, and template coordinates
        source_coo = self.affine_transform.apply_transform(
            get_det_coords(self.source_matchdets))
        template_coo = get_det_coords(self.template_matchdets)
        # Create splines describing the residual offsets in x and y left over
        # after the affine transformation
        kx = ky = spline_order
        self.sbs_x = interpolate.SmoothBivariateSpline(template_coo[:, 0],
                                                       template_coo[:, 1],
                                                       (template_coo[:, 0]
                                                        - source_coo[:, 0]),
                                                       kx=kx, ky=ky)
        self.sbs_y = interpolate.SmoothBivariateSpline(template_coo[:, 0],
                                                       template_coo[:, 1],
                                                       (template_coo[:, 1]
                                                        - source_coo[:, 1]),
                                                       kx=kx, ky=ky)

        # Make a callable to map our coordinates using these splines
        def spline_transform(xy, relative=False):
            # Returns the relative shift of xy coordinates if relative is True,
            # otherwise return the value of the transformed coordinates
            x0 = xy[0]
            y0 = xy[1]
            if relative is True:
                x0 = y0 = 0
            if xy.ndim == 2:
                xy = xy.T
            new_coo = np.array((x0 - self.sbs_x.ev(xy[0], xy[1]),
                                y0 - self.sbs_y.ev(xy[0], xy[1])))
            if xy.ndim == 2:
                return new_coo.T
            return new_coo

        self.spline_transform = spline_transform

    def align(self, hdu=None, output_filename=None, overwrite=None):
        """
        Perform the alignment and write the transformed source
        file.
        """
        if hdu is None:
            hdu = self.hdu

        if output_filename is None:
            output_filename = self.output_filename

        if overwrite is None:
            overwrite = self.overwrite

        if self.affine_transform is None:
            print("affine_transform is not defined")
            return

        source_data = self.source_fits[self.hdu].data.T

        if self.spline_transform is not None:
            def final_transform(xy):
                return (self.affine_transform.inverse().apply_transform(xy)
                        + (self.spline_transform(xy, relative=True)))
            xx, yy = np.meshgrid(np.arange(self.shape[0]),
                                 np.arange(self.shape[1]))
            spline_coords_shift = final_transform(np.array([xx, yy]))
            source_data_transform = map_coordinates(source_data,
                                                    spline_coords_shift)
        else:
            matrix, offset = self.affine_transform.inverse().matrix_form()
            source_data_transform = interpolation.affine_transform(
                source_data, matrix, offset=offset, output_shape=self.shape).T

        self.source_data_transform = source_data_transform

        if output_filename is not None:
            self.source_fits[hdu].data = source_data_transform
            self.source_fits.writeto(self.output_filename,
                                     overwrite=self.overwrite)

    def match_dets(self, transform, minmatchdist=None):
        """
        Match the source and template detections using `transform`

        Parameters
        ----------
        transform : :class:`spalipy.AffineTransform`
            The transformation to use.
        """
        if minmatchdist is None:
            minmatchdist = self.minmatchdist

        source_coo_trans = transform.apply_transform(self.source_coo)

        dists = distance.cdist(source_coo_trans, self.template_coo)
        mindists = np.min(dists, axis=1)
        passed = mindists <= minmatchdist
        sorted_idx = np.argsort(dists[passed, :])

        nmatched = np.sum(passed)
        source_matchdets = self.source_cat[passed]
        template_matchdets = self.template_cat[sorted_idx[:, 0]]

        return nmatched, source_matchdets, template_matchdets

    def trim_cat(self, cat, minfwhm=2, maxflag=4):
        """
        Trim a detection catalogue based on some SExtractor values.
        Sort this by the brightest objects then cut to the top
        `self.ndets`

        Parameters
        ----------
        minfwhm : float, optional
            The minimum value of FWHM for a valid source.
        maxflag : int, optional
            The maximum value of FLAGS for a valid source.
        """
        cat = cat[COLUMNS]
        cat = cat[(cat[FWHM] > minfwhm)
                  & (cat[FLAGS] < maxflag)]
        cat.sort(FLUX)
        cat.reverse()
        cat = cat[:self.ndets]

        return cat


class AffineTransform:
    """
    Represents an affine transformation consisting of rotation, isotropic
    scaling, and shift. [x', y'] = [[a -b], [b a]] * [x, y] + [c d]

    Parameters
    ----------
    v : tuple, list or array
        The parameters of the matrix describing the affine transformation,
        [a, b, c, d].
    """

    def __init__(self, v):
        self.v = np.asarray(v)

    def inverse(self):
        """
        Returns the inverse transform
        """
        # To represent affine transformations with matrices,
        # we can use homogeneous coordinates.
        homo = np.array([
            [self.v[0], -self.v[1], self.v[2]],
            [self.v[1],  self.v[0], self.v[3]],
            [0.0, 0.0, 1.0]
        ])
        inv = linalg.inv(homo)

        return AffineTransform((inv[0, 0], inv[1, 0], inv[0, 2], inv[1, 2]))

    def matrix_form(self):
        """
        Special output for scipy.ndimage.interpolation.affine_transform
        Returns (matrix, offset)
        """
        return (np.array([[self.v[0], -self.v[1]],
                          [self.v[1], self.v[0]]]), self.v[2:4])

    def apply_transform(self, xy):
        """
        Applies the transform to an array of x, y points
        """
        xy = np.asarray(xy)
        # Can consistently refer to x and y as xy[0] and xy[1] if xy is
        # 2D (1D coords) or 3D (2D coords) if we transpose the 2D case of xy
        if xy.ndim == 2:
            xy = xy.T
        xn = self.v[0]*xy[0] - self.v[1]*xy[1] + self.v[2]
        yn = self.v[1]*xy[0] + self.v[0]*xy[1] + self.v[3]
        if xy.ndim == 2:
            return np.column_stack((xn, yn))
        return np.stack((xn, yn))


def calc_affine_transform(source_coo, template_coo):
    """
    Calculates the affine transformation
    """
    l = len(source_coo)
    template_matrix = template_coo.ravel()
    source_matrix = np.zeros((l*2, 4))
    source_matrix[::2, :2] = np.column_stack((source_coo[:, 0],
                                              - source_coo[:, 1]))
    source_matrix[1::2, :2] = np.column_stack((source_coo[:, 1],
                                               source_coo[:, 0]))
    source_matrix[:, 2] = np.tile([1, 0], l)
    source_matrix[:, 3] = np.tile([0, 1], l)

    if l == 2:
        transform = linalg.solve(source_matrix, template_matrix)
    else:
        transform = linalg.lstsq(source_matrix, template_matrix)[0]

    return AffineTransform(transform)


def get_det_coords(cat):
    cat_arr = cat[X, Y].as_array()
    return cat_arr.view((cat_arr.dtype[0], 2))


def quad(combo, dists):
    """
    Create a hash from a combination of four dets (a "quad").

    References
    ----------
    Based on the algorithm of [L10]_.

    .. [L10] Lang, D. et al. "Astrometry.net: Blind astrometric
    calibration of arbitrary astronomical images", AJ, 2010.
    """
    max_dist_idx = np.argmax(dists)
    orders = [(0, 1, 2, 3),
              (0, 2, 1, 3),
              (0, 3, 1, 2),
              (1, 2, 0, 3),
              (1, 3, 0, 2),
              (2, 3, 0, 1)]
    order = orders[max_dist_idx]
    combo = combo[order, :]
    # Look for matrix transform [[a -b], [b a]] + [c d]
    # that brings A and B to 00 11 :
    x = combo[1, 0] - combo[0, 0]
    y = combo[1, 1] - combo[0, 1]
    b = (x-y) / (x**2 + y**2)
    a = (1/x) * (1 + b*y)
    c = b*combo[0, 1] - a*combo[0, 0]
    d = -(b*combo[0, 0] + a*combo[0, 1])

    t = AffineTransform((a, b, c, d))
    (xC, yC) = t.apply_transform((combo[2, 0], combo[2, 1])).ravel()
    (xD, yD) = t.apply_transform((combo[3, 0], combo[3, 1])).ravel()

    _hash = (xC, yC, xD, yD)
    # Break symmetries if needed
    testa = xC > xD
    testb = xC + xD > 1
    if testa:
        if testb:
            _hash = (1.0-xC, 1.0-yC, 1.0-xD, 1.0-yD)
            order = (1, 0, 2, 3)
        else:
            _hash = (xD, yD, xC, yC)
            order = (0, 1, 3, 2)
    elif testb:
        _hash = (1.0-xD, 1.0-yD, 1.0-xC, 1.0-yC)
        order = (1, 0, 3, 2)
    else:
        order = (0, 1, 2, 3)

    return combo[order, :], _hash


if __name__ == '__main__':
    def shape_type(value):
        if value is None:
            return value
        try:
            open(value)
        except FileNotFoundError:
            try:
                value = tuple(map(int, value.split(',')))
            except ValueError:
                msg = 'shape must be valid filepath or "x_size,y_size"'
                raise argparse.ArgumentTypeError(msg)
            else:
                if len(value) != 2:
                    msg = 'shape must be length-2 comma-separated'
                    raise argparse.ArgumentTypeError(msg)
        return value

    def ndets_type(value):
        try:
            return int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                msg = 'ndets must be int or a float between 0 and 1'
                raise argparse.ArgumentTypeError(msg)
            if ((value > 1) or (value <= 0)):
                msg = 'ndets as a float must be between 0 and 1'
                raise argparse.ArgumentTypeError(msg)
            return value
        return value

    parser = argparse.ArgumentParser(description='Detection-based astronomical'
                                     ' image registration.')
    parser.add_argument('source_cat', type=str, help='Filename of the source'
                        ' detection catalogue produced by SExtractor.')
    parser.add_argument('template_cat', type=str, help='Filename of the '
                        'template detection catalogue produced by SExtractor')
    parser.add_argument('source_fits', type=str, help='Filename of the source'
                        ' fits image to transform.')
    parser.add_argument('output_filename', type=str, help='Filename to write'
                        'the transformed source_fits to.')
    parser.add_argument('--shape', type=shape_type, default=None, help='Shape '
                        'of the output transformed image - either filename of '
                        'fits file to determine shape from or a "x,y" string.')
    parser.add_argument('--hdu', type=int, default=0, help='the hdu in '
                        'source_fits to transform the data of. Also the hdu '
                        'used to derive shape if shape is a fits filename.')
    parser.add_argument('--ndets', type=ndets_type, default=0.5, help='Number '
                        'of detections to use when creating quads and '
                        ' detection matching. If  0 < ndets < 1 then will '
                        'use this fraction of the shortest of source_cat and '
                        'template_cat as ndets.')
    parser.add_argument('--nquaddets', type=int, default=15, help='Number of '
                        'detections to make quads from.')
    parser.add_argument('--minquadsep', type=float, default=50, help='Minimum '
                        'disance in pixels between detections in a quad for it'
                        ' to be valid.')
    parser.add_argument('--minmatchdist', type=float, default=5, help='Minimum'
                        ' matching distance between coordinates after the '
                        'initial transformation to be considered a match.')
    parser.add_argument('--minnmatch', type=int, default=200, help='Minimum '
                        'number of matched dets for the initial '
                        'transformation to be considered sucessful.')
    parser.add_argument('--spline-order', type=int, default=3,
                        dest='spline_order', help='The order in `x` and `y` of'
                        'the spline surfaces used ' 'to correct the affine '
                        'transformation.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to '
                        'overwrite output_filename if it exists')

    args_dict = vars(parser.parse_args())

    print('Calling spalipy with:')
    print(args_dict)
    s = Spalipy(**args_dict)
    s.main()
