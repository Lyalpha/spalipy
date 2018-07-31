# spalipy

Detection-based astronomical image registration.

A slimmed-down, python 3 version of
[aplipy2.0](https://obswww.unige.ch/~tewes/alipy/) that includes an
additional warping of the transformation via splines to achieve
accurate registration in the case of non-homogeneous coordinate
transforms. This is particularly useful in the case of optically
distorted or wide field-of-view images.

## requirements

```
astropy > 1.0.0
numpy >= 1.10.0
scipy >= 1.1.0
```

Although not directly used by the program, source lists should be
provided in
[`SExtractor`](https://www.astromatic.net/software/sextractor) format.

## install

`python setup.py install`

## description

A `source` image is transformed to the pixel-coordinate system of a
`template` image using their respective detection catalogues.

Matching quads of stars between the two catalogues are used to match
detecions between the images. An affine transformation is used to best
match these coordinates using rotation, scale and translation only.

The remaining residuals between the matched detection coordinates
(after affine transformation of the `source` coordinates) are used to
construct 2D spline surfaces to represent the spatially-varying
residuals in `x` and `y` axes. These surfaces are used to calculate
the correction needed to properly register the images.

Takes approximately 25 seconds on an i7 laptop for a 50M (~8000x6000)
size source and template.


## example

Create two sextractor catalogues for our `source` and `template`

```
sex -c /path/to/my/config source.fits[0] -CATALOG_NAME source.cat
sex -c /path/to/my/config template.fits[0] -CATALOG_NAME template.cat
```
*Note that, at a minimum, the SExtracted catalogues must contain the
columns `X_IMAGE`, `Y_IMAGE`, `FLUX_BEST`, `FWHM_IMAGE`, `FLAGS`.*

Perform the alignment from the interpreter

```
s = Spalipy.spalipy("source_cat", "template_cat", "source.fits",
    		    output_filename="source-aligned.fits")
s.main()
```
*If output_filename is not specified, the transformed source data is
not written to disk but still accessible via
`s.source_data_transform`.*
*See the contents of `main()` for ways to rerun sections of the
algorithm individually, if needed - for example one could rerun
`find_spline_transform()` with a different order*
*Use `print(s.__doc__)` (or `s?` in iPython) for parameter information.*

From the command line (make a link to spalipy.py in one of your PATH
locations)

```
$ python spalipy.py source_cat template_cat source_fits source-aligned.fits
```
*output_filename is required for the command line since it doesn't make
sense to hold it in memory.*
*Use `python spalipy.py -h` for argument help.*