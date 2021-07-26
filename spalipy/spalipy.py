# spalipy - Detection-based astrononmical image registration
# Copyright (C) 2018-2021  Joe Lyman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import itertools
import logging
import os
from typing import Optional, Union

import numpy as np
import sep
from astropy.io import fits
from astropy.table import Table, vstack
from scipy import linalg, interpolate
from scipy.ndimage import interpolation, map_coordinates
from scipy.spatial import cKDTree, distance

# expose dfitpack errors so we can catch them later
try:
    interpolate.dfitpack.sproot(-1, -1, -1)
except Exception as e:
    dfitpackError = type(e)


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
        """Returns the inverse transform"""

        # To represent affine transformations with matrices,
        # we can use homogeneous coordinates.
        homo = np.array(
            [[self.v[0], -self.v[1], self.v[2]], [self.v[1], self.v[0], self.v[3]], [0.0, 0.0, 1.0]]
        )
        inv = linalg.inv(homo)

        return AffineTransform((inv[0, 0], inv[1, 0], inv[0, 2], inv[1, 2]))

    def matrix_form(self):
        """
        Special output for scipy.ndimage.interpolation.affine_transform
        Returns (matrix, offset)
        """

        return np.array([[self.v[0], -self.v[1]], [self.v[1], self.v[0]]]), self.v[2:4]

    def apply_transform(self, xy):
        """Applies the transform to an array of x, y points"""

        xy = np.asarray(xy)
        # Can consistently refer to x and y as xy[0] and xy[1] if xy is
        # 2D (1D coords) or 3D (2D coords) if we transpose the 2D case of xy
        if xy.ndim == 2:
            xy = xy.T
        xn = self.v[0] * xy[0] - self.v[1] * xy[1] + self.v[2]
        yn = self.v[1] * xy[0] + self.v[0] * xy[1] + self.v[3]
        if xy.ndim == 2:
            return np.column_stack((xn, yn))
        return np.stack((xn, yn))


class Spalipy:
    """
    Detection-based astronomical image registration.

    Parameters
    ----------
    source_data : numpy.ndarray or list
        The source image data to be transformed. Can be passed as a list of
        source images, in which case `source_det`, `source_mask` and
        `output_shape` must match in length if they are not `None`.
    source_mask : numpy.ndarray or list, optional
        The source mask data to be transformed in the same manner as
        `source_data`. This will always use nearest neighbour interpolation.
        However, the propagation of mask values is not complete, and some pixels
        in the source data that are correlated with masked pixels will not be
        masked as such in the aligned mask (particularly true for isolated masked
        pixels). If passing as a list, `None` entries can be used to indicate
        individual images without masks.
    template_data : numpy.ndarray, optional
        The template image data to which the source image will be transformed.
        Must be provided if `template_det` is `None`. If both are given then
        `template_det` takes precendence.
    source_det, template_det : None or `astropy.table.Table` or list, optional
        The detection table for the relevant image. If `None` a basic
        `sep.extract()` run will be performed to find sources. If passing
        `source_det` as a list, `None` entries can be used to indicate individual
         images where an internal extraction should be made. If `template_det`
         is `None` then `template_data` cannot be `None`.
    output_shape : None or tuple or list, optional
        The shape of the output aligned source image data. If `None`, output
        shape is the same as `source_data`. If passing as a list, `None` entries
        can be used to indicate individual images where the output shape should
        default to the same shape as the input source data.
    n_det : int or float, optional
        The number of detections to use in the alignment matching. Detections
        are sorted by the "flux" column so this will trim the detections to
        the `ndet` brightest detections in each image. If `0 < ndet <= 1`
        then `ndet` is calculated as a fractional size of the respective
        detection table.
    n_quad_det : int, optional
        The number of detections to make quads from. This will create
        `C(n_quad_det, 4) * sub_tile**2` quads, so raising the value
        too much may have a significant performance hit.
    min_quad_sep : float, optional
        Minimum distance in pixels between detections in a quad for it
        to be valid.
    max_match_dist : float, optional
        Maximum matching distance between coordinates after the
        initial transformation to be considered a match.
    min_n_match : int, optional
        Minimum number of matched dets for the affine transformation
        to be considered successful.
    sub_tile : int, optional
        Split the image into this number of sub-tiles in each axis and perform
        quad creation, affine transform fitting, and cross-matching
        independently in each sub-tile. This can help in the case of very
        large distortions where a single affine transformation will not
        describe the corner regions, for example. Set to `1` to disable this.
    max_quad_cand : int, optional
        Maximum number of quad candidates to loop through to find initial
        affine transformation.
    patience_quad_cand : int, optional
        If the affine transformation calculated from a quad does not yield
        a larger number of cross-matches than the current best transformation
        for this number of candidates, then early stop the fitting.
    max_quad_hash_dist : float, optional
        Limit on quad distances to consider a match.
    spline_order : int, optional
        The order in `x` and `y` of the final spline surfaces used to
        correct the affine transformation. If `0` then no spline
        correction is performed.
    interp_order : int, optional
        The spline order to use for interpolation - this is passed
        directly to `scipy.ndimage.affine_transform` and
        `scipy.ndimage.interpolation.map_coordinates` as the `order`
        argument. Must be in the range 0-5.
    sep_thresh : float, optional
        The threshold value to pass to `sep.extract()`.
    min_fwhm : float, optional
        The minimum value of fwhm for a valid source.
    bad_flag_bits : int, optional
        The integer representation of bad flag bits - sources matching
        at least one of these bits in their flag value will be removed.
    min_sep : float, optional
        The minimum separation between coordinates in the table, useful
        to remove crowded sources that are problem for cross-matching.
        If omitted defaults to `2 * max_match_dist`.
    skip_checking : boolean
        Whether to skip some basic checking of input arguments for
        minor speed up when setting-up. Useful when inputs are fixed
        and already known.
    """

    x_col = "x"
    y_col = "y"
    flux_col = "flux"
    fwhm_col = "fwhm"
    flag_col = "flag"
    columns = [x_col, y_col, flux_col, fwhm_col, flag_col]

    def __init__(
        self,
        source_data: Union[np.ndarray, list],
        source_mask: Union[np.ndarray, list, None] = None,
        template_data: np.ndarray = None,
        source_det: Union[Table, list, None] = None,
        template_det: Optional[Table] = None,
        output_shape: Union[tuple, list, None] = None,
        n_det: Union[float, int, None] = None,
        n_quad_det: int = 20,
        min_quad_sep: int = 50,
        max_match_dist: int = 3,
        min_n_match: int = 100,
        sub_tile: int = 1,
        max_quad_cand: int = 10,
        patience_quad_cand: int = 2,
        max_quad_hash_dist: float = 0.005,
        spline_order: int = 3,
        interp_order: int = 3,
        sep_thresh: float = 5,
        min_fwhm: float = 1,
        bad_flag_bits: int = 0,
        min_sep: float = None,
        skip_checking: bool = False,
    ):

        self.n_det = n_det
        self.n_quad_det = n_quad_det
        self.min_quad_sep = min_quad_sep
        self.max_match_dist = max_match_dist
        self.min_n_match = min_n_match
        self.sub_tile = sub_tile
        self.max_quad_cand = max_quad_cand
        self.patience_quad_cand = patience_quad_cand
        self.max_quad_hash_dist = max_quad_hash_dist
        self.spline_order = spline_order
        self.interp_order = interp_order
        self.sep_thresh = sep_thresh
        self.bad_flag_bits = bad_flag_bits
        self.min_fwhm = min_fwhm
        self.min_sep = min_sep if min_sep is not None else 2 * max_match_dist

        if isinstance(source_data, np.ndarray):
            source_data = [source_data]
            self._source_inputs_as_list = False
        else:
            self._source_inputs_as_list = True
        self._n_source_entry = len(source_data)
        if source_mask is None or isinstance(source_mask, np.ndarray):
            source_mask = [source_mask] * self._n_source_entry
        if output_shape is None or isinstance(output_shape, tuple):
            output_shape = [output_shape] * self._n_source_entry
        if source_det is None or isinstance(source_det, Table):
            source_det = [source_det] * self._n_source_entry

        if not skip_checking:
            for attr, typ, name in (
                [source_data, np.ndarray, "source_data"],
                [source_mask, np.ndarray, "source_mask"],
                [source_det, Table, "source_det"],
                [output_shape, tuple, "output_shape"],
            ):
                if not isinstance(attr, list):
                    raise ValueError(
                        f"The type of {name} was not of the correct type (expected {typ}, "
                        f"got {type(attr)}"
                    )
                else:
                    for i, _attr in enumerate(attr):
                        if not isinstance(_attr, typ) and _attr is not None:
                            raise ValueError(
                                f"The type of {name} for entry {i} was not of the correct type "
                                f"(expected one of {typ}, got {type(_attr)}"
                            )

            if len(source_data) != len(source_mask) != len(output_shape):
                raise ValueError(
                    "The lengths of source_data, source_mask and output_shape do not match"
                )
            for i, (_source_data, _source_mask) in enumerate(zip(source_data, source_mask)):
                if _source_data.ndim != 2:
                    raise ValueError(f"The dimensionality of source_data entry {i} is not 2")
                if _source_mask is not None:
                    if _source_data.shape != _source_mask.shape:
                        raise ValueError(
                            f"The shape of source_data and source_mask for entry {i} do not match"
                        )
            for i, _source_det in enumerate(source_det):
                if _source_det is not None and not isinstance(_source_det, Table):
                    raise ValueError(f"The object type for source_det entry {i} is unexpected")
            if template_data is None and template_det is None:
                raise ValueError("One of template_data or template_det must be provided")
            if template_data is not None and template_data.ndim != 2:
                raise ValueError(f"The dimensionality of template_data is not 2")

        self._source_data = source_data
        self._source_mask = source_mask
        self._source_det = []
        self._source_coo = []
        self._output_shape = []
        for i, (_source_data, _source_det, _output_shape) in enumerate(
            zip(source_data, source_det, output_shape)
        ):
            logging.info(f"Processing source entry {i}")
            det = _source_det or self._extract_detections(_source_data)
            det = self._prep_detection_table(det)
            self._source_det.append(det)
            coo = self._get_det_coords(det)
            self._source_coo.append(coo)
            shape = _output_shape or _source_data.shape
            self._output_shape.append(shape)

        self.template_data = template_data
        template_det = template_det or self._extract_detections(template_data)
        template_det = self._prep_detection_table(template_det)
        self.template_det = template_det
        self.template_coo = self._get_det_coords(self.template_det)
        self.template_coo_tree = cKDTree(self.template_coo)

        self._source_quadlist = None
        self.template_quadlist = None

        self._source_det_matched = None
        self._template_det_matched = None
        self._affine_transform = None

        self._sbs_x = None
        self._sbs_y = None
        self._spline_transform = None

        self._aligned_data = None
        self._aligned_mask = None

        self._alignment_failed = [False] * self._n_source_entry

    def _maybe_as_list(self, attr):
        """Return a property in the same form as it was passed"""
        if attr is None or self._source_inputs_as_list:
            return attr
        return attr[0]

    @property
    def source_data(self):
        return self._maybe_as_list(self._source_data)

    @property
    def source_mask(self):
        return self._maybe_as_list(self._source_mask)

    @property
    def source_det(self):
        return self._maybe_as_list(self._source_det)

    @property
    def source_coo(self):
        return self._maybe_as_list(self._source_coo)

    @property
    def source_quadlist(self):
        return self._maybe_as_list(self._source_quadlist)

    @property
    def output_shape(self):
        return self._maybe_as_list(self._output_shape)

    @property
    def source_det_matched(self):
        return self._maybe_as_list(self._source_det_matched)

    @property
    def template_det_matched(self):
        return self._maybe_as_list(self._template_det_matched)

    @property
    def affine_transform(self):
        return self._maybe_as_list(self._affine_transform)

    @property
    def sbs_x(self):
        return self._maybe_as_list(self._sbs_x)

    @property
    def sbs_y(self):
        return self._maybe_as_list(self._sbs_y)

    @property
    def spline_transform(self):
        return self._maybe_as_list(self._spline_transform)

    @property
    def aligned_data(self):
        return self._maybe_as_list(self._aligned_data)

    @property
    def aligned_mask(self):
        return self._maybe_as_list(self._aligned_mask)

    @property
    def alignment_failed(self):
        return self._maybe_as_list(self._alignment_failed)

    def align(self):
        """Performs the full alignment routine and sets resulting aligned_data attribute"""

        self.make_source_quadlist()
        self.make_template_quadlist()

        self.fit_affine_transform()
        self.fit_spline_transform()

        self.transform_data()

    def make_source_quadlist(self):
        """See `_make_quadlist()`"""
        source_quadlist = []
        for i, source_coo in enumerate(self._source_coo):
            logging.info(f"Generating source quads for entry {i}")
            source_quadlist.append(self._make_quadlist(source_coo, self._source_data[i].shape))
        self._source_quadlist = source_quadlist

    def make_template_quadlist(self):
        """See `_make_quadlist()`"""
        logging.info("Generating template quads")
        # Pass the first source_data shape here, but practically it makes no difference ..
        template_quadlist = self._make_quadlist(self.template_coo, self._source_data[0].shape)
        # .. because template quads are flattened instead of split by sub-tile
        # since we cannot be sure that image overlap means sub-tiles in the
        # source correspond to those in the template
        self.template_quadlist = [q for quadlist in template_quadlist for q in quadlist]

    def fit_affine_transform(self):
        """
        Calls fit_affine_transform for a list of source entries
        """
        source_det_matched = []
        template_det_matched = []
        affine_transform = []
        for i in range(len(self._source_data)):
            logging.info(f"Fitting affine transform and cross-matching for entry {i}")
            (
                _source_det_matched,
                _template_matched_det,
                _affine_transform,
            ) = self._fit_affine_transform(i)
            source_det_matched.append(_source_det_matched)
            template_det_matched.append(_template_matched_det)
            affine_transform.append(_affine_transform)
        if all(self._alignment_failed):
            raise RuntimeError("No source entries sucessfully found an affine transform")

        self._source_det_matched = source_det_matched
        self._template_det_matched = template_det_matched
        self._affine_transform = affine_transform

    def _fit_affine_transform(self, entry):
        """
        Use the quadlist hashes to determine an initial guess at an affine
        transformation and determine matched detections lists. Then refine
        the transformation using the matched detection lists.

        entry identifies the index number of the source entry being processed

        Returns the source and template matched detections, and the affine transform.
        """
        # Template quads are always flattened
        template_hash = np.array([q[1] for q in self.template_quadlist])

        n_match_full = 0
        source_det_matched_full = []
        template_det_matched_full = []
        # Iterate over each sub-tile in the source_data and perform cross-matching
        # using a per-sub-tile transform
        entry_shape = self._source_data[entry].shape
        for i, (source_quadlist, source_det) in enumerate(
            zip(
                self._source_quadlist[entry],
                self._sub_tile_det(self._source_det[entry], entry_shape),
            ),
            1,
        ):
            logging.debug(f"Processing sub-tile region {i}/{self.sub_tile ** 2}")
            source_hash = np.array([q[1] for q in source_quadlist])
            dists = distance.cdist(template_hash, source_hash)
            min_dist_idx = np.argmin(dists, axis=0)
            min_dist = np.min(dists, axis=0)
            best = np.argsort(min_dist)
            if not np.any(min_dist < self.max_quad_hash_dist):
                logging.warning(
                    f"No matching quads found below minimum quad hash distance of {self.max_quad_hash_dist}"
                )
                continue

            patience = 0
            best_n_match = 0
            source_det_matched = []
            template_det_matched = []
            n_quad_cand = min(self.max_quad_cand, len(best))

            for j in range(n_quad_cand):
                logging.debug(
                    f"Running affine transformation fit on quad candidate {j + 1}/{n_quad_cand}"
                )
                # Use best initial guess at transformation to get list of matched detections
                bi = best[j]
                dist = min_dist[bi]
                if dist < self.max_quad_hash_dist:
                    # Get a quick (exact) transformation guess using first two detections
                    template_quad = self.template_quadlist[min_dist_idx[bi]]
                    source_quad = source_quadlist[bi]

                    initial_affine_transform = calc_affine_transform(
                        source_quad[0][:2], template_quad[0][:2]
                    )
                    n_match, _source_det_matched, _template_det_matched = self._match_dets(
                        source_det, initial_affine_transform
                    )

                    if n_match:
                        # Refine the transformation using the coordinates of the matched detections
                        source_match_coo = self._get_det_coords(_source_det_matched)
                        template_match_coo = self._get_det_coords(_template_det_matched)
                        _affine_transform = calc_affine_transform(
                            source_match_coo, template_match_coo
                        )
                        n_match, _source_det_matched, _template_det_matched = self._match_dets(
                            source_det, initial_affine_transform
                        )

                        if n_match > best_n_match:
                            logging.info(
                                f"Found new best number of matched detections ({n_match}) - updating transform"
                            )
                            source_det_matched = _source_det_matched
                            template_det_matched = _template_det_matched
                            best_n_match = n_match
                            patience = 0
                        else:
                            patience += 1
                        if patience == self.patience_quad_cand:
                            logging.info(
                                f"No improvement in cross-match performance found in "
                                f"{self.patience_quad_cand} iterations - exiting"
                            )
                            break
            n_match_full += best_n_match
            source_det_matched_full.append(source_det_matched)
            template_det_matched_full.append(template_det_matched)

        if n_match_full < self.min_n_match:
            logging.error(
                f"Number of affine transform matched detections "
                f"({n_match_full}) < minumum required ({self.min_n_match})"
            )
            logging.error(f"Alignment failed for entry {entry}")
            self._alignment_failed[entry] = True
            return None, None, None
        logging.info(
            f"Matched {n_match_full} detections within {self.max_match_dist} "
            f"pixels with initial affine transformation(s)"
        )
        self._alignment_failed[entry] = False

        # Convert list of tables from sub-tiles into a single table
        source_det_matched = vstack(source_det_matched_full)
        template_det_matched = vstack(template_det_matched_full)

        # Now use the total cross-matches from all sub-tiles to calculate an overall affine
        # transformation
        logging.info("Generating overall affine transform")
        source_match_coo = self._get_det_coords(source_det_matched)
        template_match_coo = self._get_det_coords(template_det_matched)
        affine_transform = calc_affine_transform(source_match_coo, template_match_coo)

        return source_det_matched, template_det_matched, affine_transform

    def fit_spline_transform(self):
        """
        Calls _fit_spline_transform for a list of source entries
        """
        if self.spline_order == 0:
            logging.info("Skipping spline transformation fit (spline order is 0)")
            return

        sbs_x = []
        sbs_y = []
        spline_transform = []
        for i in range(len(self._source_data)):
            if self._alignment_failed[i]:
                logging.info(f"Skipping spline transform for entry {i} due to failed alignment")
                _sbs_x, _sbs_y, _spline_transform = None, None, None
            else:
                logging.info(f"Fitting spline transform for entry {i}")
                _sbs_x, _sbs_y, _spline_transform = self._fit_spline_transform(i)
            sbs_x.append(_sbs_x)
            sbs_y.append(_sbs_y)
            spline_transform.append(_spline_transform)
        if all(self._alignment_failed):
            raise RuntimeError("No source entries sucessfully found a spline transform")

        self._sbs_x = sbs_x
        self._sbs_y = sbs_y
        self._spline_transform = spline_transform

    def _fit_spline_transform(self, entry):
        """
        Determine the residual `x` and `y` offsets between matched coordinates
        after affine transformation and fit 2D spline surfaces to describe the
        spatially-varying correction to be applied.

        entry identifies the index number of the source entry being processed.

        Returns the smooth bivariate spline objects in x and y directions,
        and the spline transform callable to transofrm coordinates.
        """

        # Get the source, after affine transformation, and template coordinates
        source_coo = self._get_det_coords(self._source_det_matched[entry])
        source_coo = self._affine_transform[entry].apply_transform(source_coo)
        template_coo = self._get_det_coords(self._template_det_matched[entry])

        # Create splines describing the residual offsets in x and y left over
        # after the affine transformation
        kx = ky = self.spline_order
        try:
            sbs_x = interpolate.SmoothBivariateSpline(
                template_coo[:, 0],
                template_coo[:, 1],
                (template_coo[:, 0] - source_coo[:, 0]),
                kx=kx,
                ky=ky,
            )
            sbs_y = interpolate.SmoothBivariateSpline(
                template_coo[:, 0],
                template_coo[:, 1],
                (template_coo[:, 1] - source_coo[:, 1]),
                kx=kx,
                ky=ky,
            )
        except dfitpackError:
            logging.error(
                "scipy.interpolate.SmoothBivariateSpline raised dfitpackError (not enough sources?)"
            )
            logging.error(f"Alignment failed for entry {entry}")
            self._alignment_failed[entry] = True
            return None, None, None

        # Make a callable to map our coordinates using these splines
        def spline_transform(xy, relative=False):
            """
            Return x,y coordinates or shifts after spline transformation

            If `relative = True` return relative shift of x,y coordinates,
            otherwise return the absolute pixel value of the transformed
            coordinates
            """
            if xy.ndim == 2:
                xy = xy.T
            x0 = xy[0]
            y0 = xy[1]
            if relative is True:
                x0 = y0 = 0
            spline_x_offsets = sbs_x.ev(xy[0], xy[1])
            spline_y_offsets = sbs_y.ev(xy[0], xy[1])
            new_coo = np.array((x0 - spline_x_offsets, y0 - spline_y_offsets))
            if xy.ndim == 2:
                return new_coo.T
            return new_coo

        self._alignment_failed[entry] = False
        return sbs_x, sbs_y, spline_transform

    def full_transform(self, coo, entry, inverse=True):
        """Return transformed coordinates including both affine and spline transforms"""
        if inverse:
            ft = self._affine_transform[entry].inverse().apply_transform(coo)
            if self._spline_transform is not None and self._spline_transform[entry] is not None:
                ft = ft + self._spline_transform[entry](coo, relative=True)
        else:
            ft = self._affine_transform[entry].apply_transform(coo)
            if self._spline_transform is not None and self._spline_transform[entry] is not None:
                ft = ft - self._spline_transform[entry](coo, relative=True)
        return ft

    def transform_data(self):
        """
        Calls _transform_data for a list of source entries
        """
        aligned_data = []
        aligned_mask = []
        for i in range(len(self._source_data)):
            if self._alignment_failed[i]:
                logging.info(f"Skipping spline transform for entry {i} due to failed alignment")
                _aligned_data, _aligned_mask = None, None
            else:
                logging.info(f"Aligning source data entry {i}")
                _aligned_data, _aligned_mask = self._transform_data(i)
            aligned_data.append(_aligned_data)
            aligned_mask.append(_aligned_mask)

        self._aligned_data = aligned_data
        self._aligned_mask = aligned_mask

    def _transform_data(self, entry):
        """
        Perform the alignment and return aligned data and mask

        entry identifies the index number of the source entry being processed.

        Returns the aligned data and mask (will be None if no source mask for that
        source entry).
        """
        aligned_mask = None
        if self._spline_transform is not None and self._spline_transform[entry] is not None:
            logging.info("Applying affine + spline transformation to source_data")
            xx, yy = np.meshgrid(
                np.arange(self._output_shape[entry][0]), np.arange(self._output_shape[entry][1])
            )
            full_transform_coords_shift = self.full_transform(np.array([xx, yy]), entry)
            aligned_data = map_coordinates(
                self._source_data[entry].T, full_transform_coords_shift, order=self.interp_order
            )
            if self._source_mask[entry] is not None:
                aligned_mask = map_coordinates(
                    self._source_mask[entry].T, full_transform_coords_shift, order=0
                )
        else:
            logging.info("Applying affine transformation to source_data")
            matrix, offset = self._affine_transform[entry].inverse().matrix_form()
            aligned_data = interpolation.affine_transform(
                self._source_data[entry].T,
                matrix,
                offset=offset,
                order=self.interp_order,
                output_shape=self._output_shape[entry][::-1],
            ).T
            if self._source_mask[entry] is not None:
                aligned_mask = interpolation.affine_transform(
                    self._source_mask[entry].T,
                    matrix,
                    offset=offset,
                    order=0,
                    output_shape=self._output_shape[entry][::-1],
                ).T

        return aligned_data, aligned_mask

    def log_transform_stats(self):
        for i in range(len(self._source_data)):
            logging.info(f"\nSource entry {i}:")
            n_match, dx_mean, dx_med, dx_std, dy_mean, dy_med, dy_std = self._residuals(i)
            logging.info(f"Matched {n_match} detections within {self.max_match_dist} pixels ")
            logging.info("Pixel residuals between matched detections [mean, median (stddev)]:")
            logging.info(f"x = {dx_mean:.3f}, {dx_med:.3f} ({dx_std:.3f})")
            logging.info(f"y = {dy_mean:.3f}, {dy_med:.3f} ({dy_std:.3f})")

    def _get_det_coords(self, det):
        """Return 2D array of x, y pixel coordinates from a detection table"""
        cat_arr = det[self.x_col, self.y_col].as_array()
        return cat_arr.view((cat_arr.dtype[0], 2))

    def _sub_tile_coo(self, coo, shape):
        """Return a generator of coordinates in each sub-tile"""
        for sub_tile_mask in self._sub_tile_mask(coo, shape):
            yield coo[sub_tile_mask]

    def _sub_tile_det(self, det, shape):
        """Return a generator of detections in each sub-tile"""
        coo = self._get_det_coords(det)
        for sub_tile_mask in self._sub_tile_mask(coo, shape):
            yield det[sub_tile_mask]

    def _sub_tile_mask(self, coo, shape):
        """Return a generator of masks describing membership of sub-tiles"""
        if self.sub_tile == 1:
            yield slice(None)
        else:
            width = shape[0]
            height = shape[1]
            sub_tile_width = width / self.sub_tile
            sub_tile_height = height / self.sub_tile
            for i in range(self.sub_tile):
                sub_tile_centre_x = width * (2 * i + 1) / (self.sub_tile * 2)
                for j in range(self.sub_tile):
                    sub_tile_centre_y = height * (2 * j + 1) / (self.sub_tile * 2)
                    sub_tile_mask = (
                        np.abs(sub_tile_centre_x - coo[:, 0]) <= (sub_tile_width / 2)
                    ) & (np.abs(sub_tile_centre_y - coo[:, 1]) <= (sub_tile_height / 2))
                    yield sub_tile_mask

    def _extract_detections(self, data):
        """Return an astropy Table of detections found in input data"""

        try:
            bkg = sep.Background(data)
        except ValueError:
            # See https://sep.readthedocs.io/en/latest/tutorial.html#Finally-a-brief-word-on-byte-order
            data = data.byteswap(inplace=True).newbyteorder()
            bkg = sep.Background(data)
        bkg_rms = bkg.rms()
        data_sub = data - bkg.back()

        extracted_det = sep.extract(data_sub, thresh=self.sep_thresh, err=bkg_rms,)
        det = Table(extracted_det)
        det["fwhm"] = 2.0 * (np.log(2) * (det["a"] ** 2.0 + det["b"] ** 2.0)) ** 0.5
        logging.info(f"Initially extracted {len(det)} detections")

        return det

    def _prep_detection_table(self, det):
        """Filter detection table to a subset of rows and columns to be used in alignment"""

        # Keep only important columns
        det = det[self.columns]
        det = det[det[self.fwhm_col] >= self.min_fwhm]
        det = det[(det[self.flag_col] & self.bad_flag_bits) == 0]
        det.sort(self.flux_col, reverse=True)

        if self.min_sep > 0:
            tree = cKDTree(self._get_det_coords(det))
            close_pairs = tree.query_pairs(self.min_sep)
            to_remove = [det for pair in close_pairs for det in pair]
            if to_remove:
                det.remove_rows(np.unique(to_remove))

        if isinstance(self.n_det, float):
            n_det = int(self.n_det * len(det))
        else:
            n_det = self.n_det

        if self.n_det is not None:
            det = det[:n_det]
        logging.info(f"Using {len(det)} detections in alignment")

        if len(det) < self.min_n_match:
            raise ValueError(
                f"Number of detections found is less than min_n_match ({self.min_n_match}) "
                "- no solution possible"
            )

        return det

    def _make_quadlist(self, coo, shape):
        """
        Create hashes for "quads" of detection coordinates

        The quads are returned in a list of lists, where each parent list
        represents the quads made in a separate sub_tile of the image.

        shape is used when iterating over subtiles.
        """

        full_quadlist = []
        for _coo in self._sub_tile_coo(coo, shape):
            if self.n_quad_det > len(_coo):
                logging.warning(
                    f"Low number of detections found - restricting number of "
                    f"detections used for quads from {self.n_quad_det} to {len(_coo)}"
                )
                n_quad_det = len(_coo)
            else:
                n_quad_det = self.n_quad_det
            subtile_quadlist = []
            quad_idxs = itertools.combinations(range(n_quad_det), 4)
            for quad_idx in quad_idxs:
                four_detections_coo = _coo[quad_idx, :]
                four_detections_dist = distance.pdist(four_detections_coo)
                if np.min(four_detections_dist) > self.min_quad_sep:
                    subtile_quadlist.append(quad(four_detections_coo, four_detections_dist))
            full_quadlist.append(subtile_quadlist)

        return full_quadlist

    def _match_dets(
        self, source_det: Table, transform: Optional[AffineTransform] = None, entry=None
    ):
        """
        Match detections between source and template tables.

        Will use the class instance's `full_transform()` method to transform
        source coordinates. Otherwise an affine transform can be passed to use
        that only. `source_det` can be a subset of the total source detections
        (for example when employing sub-tiling), but the full template
        detections are always used to match to since we cannot be sure of
        source-template overlap.
        """
        source_coo = self._get_det_coords(source_det)
        if transform is not None:
            source_coo_trans = transform.apply_transform(source_coo)
        else:
            source_coo_trans = self.full_transform(source_coo, entry, inverse=False)
        source_matched_mask, template_matched_idxs = self._match_coo(source_coo_trans)
        source_det_matched = source_det[source_matched_mask]
        template_det_matched = self.template_det[template_matched_idxs]
        return np.sum(source_matched_mask), source_det_matched, template_det_matched

    def _match_coo(self, source_coo_trans: np.ndarray):
        """Match transformed source and template coordinates"""

        # Obtain the two nearest neighbours matches in the template to
        # the transformed source coordinates. The upper bound means that
        # any beyond this distance are filled with inf values.
        dists, idxs = self.template_coo_tree.query(
            source_coo_trans, k=2, distance_upper_bound=2 * self.max_match_dist
        )
        # For a match, we require the match distance to be within our limit, and
        # that the second nearest object is at least double that distance. This is
        # a crude method to alleviate double matches, maybe caused by aggressive
        # segmentation in the source catalogues.
        source_matched_mask = (dists[:, 0] < self.max_match_dist) & np.isinf(dists[:, 1])
        template_matched_idxs = idxs[source_matched_mask, 0]

        return source_matched_mask, template_matched_idxs

    def _residuals(self, entry):
        """Returns statistics for residual offsets, after transformation"""

        n_match, source_det_matched, template_det_matched = self._match_dets(
            self._source_det[entry], entry=entry
        )
        source_coo = self._get_det_coords(self._source_det_matched[entry])
        source_coo_trans = self.full_transform(source_coo, entry, inverse=False)
        template_coo = self._get_det_coords(self._template_det_matched[entry])

        dx = template_coo[:, 0] - source_coo_trans[:, 0]
        dy = template_coo[:, 1] - source_coo_trans[:, 1]

        return (
            n_match,
            np.mean(dx),
            np.median(dx),
            np.std(dx),
            np.mean(dy),
            np.median(dy),
            np.std(dy),
        )


def calc_affine_transform(source_coo, template_coo):
    """Calculate an affine transformation between two coordinate sets"""

    n = len(source_coo)
    template_matrix = template_coo.ravel()
    source_matrix = np.zeros((n * 2, 4))
    source_matrix[::2, :2] = np.column_stack((source_coo[:, 0], -source_coo[:, 1]))
    source_matrix[1::2, :2] = np.column_stack((source_coo[:, 1], source_coo[:, 0]))
    source_matrix[:, 2] = np.tile([1, 0], n)
    source_matrix[:, 3] = np.tile([0, 1], n)

    if n == 2:
        transform = linalg.solve(source_matrix, template_matrix)
    else:
        transform = linalg.lstsq(source_matrix, template_matrix)[0]

    return AffineTransform(transform)


def quad(coo, dist):
    """
    Create a hash from a combination of four detections coordinates and distances (a "quad").

    Returns
    -------
    coo : numpy.ndarray
        The four detection coordinates in the correct (expected) quad order.
    hash : tuple
        A hash sequence describing the quad.

    References
    ----------
    Based on the algorithm of [L10]_.

    .. [L10] Lang, D. et al. "Astrometry.net: Blind astrometric
    calibration of arbitrary astronomical images", AJ, 2010.
    """

    max_dist_idx = int(np.argmax(dist))
    orders = [
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ]
    order = orders[max_dist_idx]
    coo = coo[order, :]
    # Look for matrix transform [[a -b], [b a]] + [c d]
    # that brings A and B to 00 11 :
    x = coo[1, 0] - coo[0, 0]
    y = coo[1, 1] - coo[0, 1]
    b = (x - y) / (x ** 2 + y ** 2)
    a = (1 / x) * (1 + b * y)
    c = b * coo[0, 1] - a * coo[0, 0]
    d = -(b * coo[0, 0] + a * coo[0, 1])

    t = AffineTransform((a, b, c, d))
    (xC, yC) = t.apply_transform((coo[2, 0], coo[2, 1])).ravel()
    (xD, yD) = t.apply_transform((coo[3, 0], coo[3, 1])).ravel()

    _hash = (xC, yC, xD, yD)
    # Break symmetries if needed
    testa = xC > xD
    testb = xC + xD > 1
    if testa:
        if testb:
            _hash = (1.0 - xC, 1.0 - yC, 1.0 - xD, 1.0 - yD)
            order = (1, 0, 2, 3)
        else:
            _hash = (xD, yD, xC, yC)
            order = (0, 1, 3, 2)
    elif testb:
        _hash = (1.0 - xD, 1.0 - yD, 1.0 - xC, 1.0 - yC)
        order = (1, 0, 3, 2)
    else:
        order = (0, 1, 2, 3)

    return coo[order, :], _hash


def _read_sextractor_cat(cat_filename):
    cat = Table.read(cat_filename, format="ascii.sextractor")
    sex_columns = ["X_IMAGE", "Y_IMAGE", "FLUX", "FWHM_IMAGE", "FLAGS"]
    spalipy_columns = [
        Spalipy.x_col,
        Spalipy.y_col,
        Spalipy.flux_col,
        Spalipy.fwhm_col,
        Spalipy.flag_col,
    ]
    for sex_col, spalipy_col in zip(sex_columns, spalipy_columns):
        cat.rename_column(sex_col, spalipy_col)
    return cat


def _console_align(args_dict):
    """Performs the alignment routine based on command-line arguments."""

    # Set up logging for command-line usage
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(len(log_levels) - 1, args_dict["verbose"])]
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=log_level)

    logging.info(f"Reading source data from {args_dict['source_fits']}")
    args_dict["source_data"] = fits.getdata(args_dict["source_fits"], ext=args_dict["source_ext"])
    if args_dict.get("source_cat", None) is not None:
        logging.info(f"Reading source detection catalogue from {args_dict['source_cat']}")
        args_dict["source_det"] = _read_sextractor_cat(args_dict["source_cat"])

    if args_dict["template_fits"] is not None:
        logging.info(f"Reading template data from {args_dict['template_fits']}")
        args_dict["template_data"] = fits.getdata(
            args_dict["template_fits"], ext=args_dict["template_ext"]
        )
    else:
        logging.info(f"Reading template detection catalogue from {args_dict['template_cat']}")
        args_dict["template_det"] = _read_sextractor_cat(args_dict["template_cat"])

    # Sore the output filename and overwrite flag for use later
    output_filename = args_dict.pop("output_filename")
    overwrite = args_dict.pop("overwrite")
    # Check whether there will be problems writing the file later
    if not overwrite and os.path.exists(output_filename):
        raise ValueError(
            f"Output filename {output_filename} exists and overwrite flag was not given"
        )

    # Remove any args not used in the call to Spalipy
    args_dict.pop("source_fits", None)
    args_dict.pop("template_fits", None)
    args_dict.pop("source_cat", None)
    args_dict.pop("template_cat", None)
    args_dict.pop("source_ext", None)
    args_dict.pop("template_ext", None)
    args_dict.pop("verbose", None)

    logging.info("Initialising Spalipy instance")
    s = Spalipy(**args_dict)
    s.align()
    s.log_transform_stats()

    logging.info(f"Writing aligned source data to {output_filename}")
    fits.writeto(output_filename, data=s.aligned_data, overwrite=overwrite)


def main(args=None):
    """The function called by align-fits script."""

    def shape_type(value):
        if value is None:
            return value
        try:
            value = tuple(map(int, value.split(",")))
            if len(value) != 2:
                raise ValueError
        except ValueError:
            msg = 'shape must be of the format "x_size,y_size"'
            raise argparse.ArgumentTypeError(msg)
        return value

    def ndet_type(value):
        try:
            return int(value)
        except ValueError:
            try:
                value = float(value)
                if (value > 1) or (value <= 0):
                    raise ValueError
            except ValueError:
                msg = "ndet must be integer or a float between 0 and 1"
                raise argparse.ArgumentTypeError(msg)
            return value

    parser = argparse.ArgumentParser(
        description="Detection-based astronomical image registration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "source_fits", type=str, help="Filename of the source fits image to transform.",
    )

    parser.add_argument(
        "output_filename", type=str, help="Filename to write the transformed source image to.",
    )

    parser.add_argument(
        "-sc",
        "--source-cat",
        type=str,
        default=None,
        help="Filename of the source image detection catalogue produced by "
        "SExtractor. This *must* include the parameters X_IMAGE, Y_IMAGE, "
        "FLUX_BEST, FWHM_IMAGE and FLAGS. If omitted an internal run of "
        "`sep` will extract detections from the image data.",
    )

    template_group = parser.add_mutually_exclusive_group(required=True)
    template_group.add_argument(
        "-tf",
        "--template-fits",
        type=str,
        default=None,
        help="Filename of the template fits image to transform to. If "
        "provided then an internal run of `sep` will extract detections from "
        "the image data. Only one of `template_fits` and `template_cat` "
        "should be provided.",
    )
    template_group.add_argument(
        "-tc",
        "--template-cat",
        type=str,
        default=None,
        help="Filename of the template image detection catalogue produced by "
        "SExtractor. This *must* include the parameters X_IMAGE, Y_IMAGE, "
        "FLUX_BEST, FWHM_IMAGE and FLAGS. Only one of `template_fits` and "
        "`template_cat` should be provided.",
    )

    parser.add_argument(
        "--output-shape",
        type=shape_type,
        default=None,
        help="Shape of the output transformed source image in the form "
        "'x_size,y_size'. If omitted will take the same shape as the "
        "input source image.",
    )
    parser.add_argument(
        "--source-ext",
        type=int,
        default=0,
        help="The hdu extension in `source_fits` to get the image data from.",
    )
    parser.add_argument(
        "--template-ext",
        type=int,
        default=0,
        help="The hdu extension in `template_fits` to get the image data from.",
    )
    parser.add_argument(
        "--n-det",
        type=ndet_type,
        default=0.25,
        help="The number of detections to use in the alignment matching. "
        "Detections are sorted by the 'flux' column, so this will trim the "
        "detections to the `ndet` brightest detections in each image. If "
        "`0 < ndet <= 1` then `ndet` is calculated as a fractional size of "
        "the source detection catalogue.",
    )
    parser.add_argument(
        "--n-quad-det",
        type=int,
        default=20,
        help="The number of detections to make quads from. This will create "
        "`C(n_quad_det, 4) * sub_tile**2` quads, so raising the value too "
        "much may have a significant performance hit.",
    )
    parser.add_argument(
        "--min-quad-sep",
        type=float,
        default=50,
        help="Minimum disance in pixels between detections in a quad for it to be valid.",
    )
    parser.add_argument(
        "--max-match-dist",
        type=float,
        default=3,
        help="Minimum matching distance between coordinates after the initial transformation to be considered a match.",
    )
    parser.add_argument(
        "--min-n-match",
        type=int,
        default=100,
        help="Minimum number of matched dets for the affine transformation(s) to be considered sucessful.",
    )
    parser.add_argument(
        "--sub-tile",
        type=int,
        default=1,
        help="Split the image into this number of sub-tiles in each axis and "
        "perform quad creation, affine transform fitting, and cross-matching "
        "independently in each sub-tile. This can help in the case of very "
        "large distortions where a single affine transformation will not "
        "describe the corner regions, for example. Set to `1` to disable "
        "this.",
    )
    parser.add_argument(
        "--max-quad-cand",
        type=int,
        default=10,
        help="Maximum number of quad candidates to loop through to find "
        "initial affine transformation.",
    )
    parser.add_argument(
        "--patience-quad-cand",
        type=int,
        default=2,
        help="If the affine transformation calculated from a quad does not "
        "yield a larger number of cross-matches than the current best "
        "transformation for this number of candidates, then early stop "
        "the fitting.",
    )
    parser.add_argument(
        "--max-quad-hash-dist",
        type=float,
        default=0.005,
        help="Limit on quad distances to consider a match.",
    )
    parser.add_argument(
        "--spline-order",
        type=int,
        default=3,
        help="The order in `x` and `y` of the spline surfaces used "
        "to correct the affine transformation. If `0` then no spline "
        "correction is performed.",
    )
    parser.add_argument(
        "--interp-order",
        type=int,
        default=3,
        help="The spline order to use for interpolation - this is passed "
        "directly to `scipy.ndimage.affine_transform` and"
        "`scipy.ndimage.interpolation.map_coordinates` as the `order`"
        "argument. Must be in the range 0-5.",
    )
    parser.add_argument(
        "--sep-thresh",
        type=float,
        default=5,
        help="The threshold value to pass to `sep.extract()`.",
    )
    parser.add_argument(
        "--min-fwhm", type=float, default=1, help="The minimum value of fwhm for a valid source.",
    )
    parser.add_argument(
        "--bad-flag-bits",
        type=int,
        default=0,
        help="The integer representation of bad flag bits - sources matching "
        "at least one of these bits in their flag value will be removed.",
    )
    parser.add_argument(
        "--min-sep",
        type=float,
        default=0,
        help="The minimum separation between coordinates in the table, "
        "useful to remove crowded sources that are problem for "
        "cross-matching. If omitted defaults to `2 * max_match_dist`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite output_filename if it exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of logging. `-v` for INFO messages, "
        "`-vv` for DEBUG. Default is WARNING.",
    )

    args_dict = vars(parser.parse_args(args))

    _console_align(args_dict)


def main_simple(args=None):
    """The function called by align-fits-simple script."""
    parser = argparse.ArgumentParser(
        description="Detection-based astronomical image registration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "source_fits", type=str, help="Filename of the source fits image to transform.",
    )

    parser.add_argument(
        "output_filename", type=str, help="Filename to write the transformed source image to.",
    )
    parser.add_argument(
        "template_fits",
        type=str,
        default=None,
        help="Filename of the template fits image to transform to.",
    )
    parser.add_argument(
        "--source-ext",
        type=int,
        default=0,
        help="The hdu extension in `source_fits` to get the image data from.",
    )
    parser.add_argument(
        "--template-ext",
        type=int,
        default=0,
        help="The hdu extension in `template_fits` to get the image data from.",
    )
    parser.add_argument(
        "--min-n-match",
        type=int,
        default=100,
        help="Minimum number of matched dets for the affine transformation "
        "to be considered sucessful.",
    )
    parser.add_argument(
        "--sep-thresh",
        type=float,
        default=5,
        help="The threshold value to pass to `sep.extract()`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite output_filename if it exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of logging. `-v` for INFO messages, "
        "`-vv` for DEBUG. Default is WARNING.",
    )

    args_dict = vars(parser.parse_args(args))

    args_dict["n_det"] = 0.9
    args_dict["sub_tile"] = 1
    args_dict["patience_quad_cand"] = 1
    args_dict["spline_order"] = 0

    _console_align(args_dict)
