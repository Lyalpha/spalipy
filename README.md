# spalipy

Detection-based astronomical image registration.

Initially built from the algorithm of
[alipy2.0](https://obswww.unige.ch/~tewes/alipy/), `spalipy` includes an
optional additional warping of the affine transformation via splines to 
achieve accurate registration in the case of non-homogeneous coordinate 
transforms. This is particularly useful in the case of optically distorted 
or wide field-of-view images.

## Install

#### From PyPI
```
pip install spalipy
```

#### From source
```
git clone https://github.com/Lyalpha/spalipy
cd spalipy
pip install .
```

## Quick run

If you have (geometrically) well-behaved images with a significant
overlap, then good results can usually be obtained with a call such
as:
```
align-fits-simple source.fits source_aligned.fits template.fits
```

To take advantage of all the dials and sliders to tweak the alignment,
take a look at the entire parameter descriptions via:
```
align-fits -h
```

Alternatively, one can pass lower level objects to perform an alignment
interactively or within an external script, see running `spalipy` 
[interactively](#interactively).

## Description

A `source` image is transformed to the pixel-coordinate system of a
`template` image using their respective detections as tie-points.

Matching [quads](https://arxiv.org/abs/0910.2233) of detections between 
the two catalogues are used to match corresponding detections in the two 
images. An initial affine transformation is calculated from a quad match,
and is applied to `source` image detections. Following this, cross-matching
is performed within some tolerance to find corresponding detections across
the image. The remaining residuals between the matched detection coordinates 
are used to construct 2D spline surfaces that represent the spatially-varying 
residuals in `x` and `y` axes. These surfaces are used to calculate the 
correction needed to properly register the images even in the face of 
non-homogeneous coordinate transformation between the images. Flux 
conservation is relatively robust so long as the pixel scale between `source`
and `template` is the same. Proper investigation with different pixel scales
has not been performed.

*__Note:__ the affine transformation uses `scipy.interpolation.affine_transform`
which doesn't handle nans properly, therefore replace all nan values
in the `source` image prior to running `spalipy`.*





## Examples

`spalipy` can be run in two modes - via the command-line scripts or 
interactively. The second big choice is to either provide your own detection
catalogues or let `spalipy` perform its own detection. Each of these scenarios
is shown below.

### via the command-line

##### Using the internal detection routine

When using the internal detection routines, there are two command-line 
scripts: `align-fits` and `align-fits-simple`. For narrow field-of-view
images without significant distortions, `align-fits-simple` is probabably
entirely sufficient to get a good alignment. (`align-fits-simple` has
a significantly reduced parameter list and sets some automatically,
for example it will always switch off spline fitting and does not
allow the user to pass existing detection catalogues.)

```
align-fits-simple source.fits source_aligned.fits template.fits
```
or
```
align-fits source.fits source_aligned.fits -tf template.fits
```
Take notice of the `-tf` argument in the second example, this is because 
`align-fits` offers multiple ways to provide detections, as shown in the next
section.

##### Passing existing SExtractor detection catalogues

If one already has detection catalogues from a SExtractor run, then these can
be used to save repetition.

e.g. create two `SExtractor` catalogues for the image:
```
sex -c /path/to/my/sex.config source.fits -CATALOG_NAME source.cat
sex -c /path/to/my/sex.config template.fits -CATALOG_NAME template.cat
```
*__Note:__ At a minimum, the `SExtractor` catalogues __must__ contain the 
columns `X_IMAGE, Y_IMAGE, FLUX, FWHM_IMAGE, FLAGS`.*

We must use `align-fits` here since `align-fits-simple` does not allow us to
pass catalogues:

```
align-fits source.fits source_aligned.fits -sc source.cat -tc template.cat
```

### Interactively

##### Using the internal detection routine

```python
from astropy.io import fits
from spalipy import Spalipy

source_data = fits.getdata("source.fits")
template_data = fits.getdata("template.fits")

sp = Spalipy(source_data, template_data=template_data)
sp.align()
fits.writeto("source_aligned.fits", data=sp.aligned_data)
```

##### Passing existing detection tables

Analagously to [passing `SExtractor` catalogues](#passing-existing-sextractor-detection-catalogues),
one can pass existing `astropy.Table` objects when calling `spalipy` interactively, for examples
as the output of a prior `sep.extract()` call.

*__Note:__ At a minimum, the detection tables __must__ contain the 
columns `x, y, flux, fwhm, flag`.*

```python
import sep
from astropy.io import fits
from astropy.table import Table
from spalipy import Spalipy

source_data = fits.getdata("source.fits")
template_data = fits.getdata("template.fits")
# Run sep on each set of data
# ...
# source_extracted = sep.extract(...)
# template_extracted = sep.extract(...)
source_det = Table(source_extracted)
template_det = Table(template_extracted)

sp = Spalipy(source_data, source_det=source_det, template_det=template_det)
sp.align()
fits.writeto("source_aligned.fits", data=sp.aligned_data)
```

### Logging

When running interactively, all information is output in logging. To see these
one can do
```python
import logging
logging.getLogger().setLevel(logging.INFO)  # or logging.DEBUG for more messages
```
prior to any of the [interactive example calls](#interactively).

Statistics for the transformation goodness can also be accessed via:
```python
sp.log_transform_stats()
```

### Parameter tuning

Several parameters should have the main focus of attention if an acceptable 
alignment is not being found.

* If you have a small number of detections in your image overlap then
`min_n_match` will need to be lowered from its default of `100`, but it is
also worth raising `n_det` so that the alignment uses all of your sources.
See `n_det` docstring on its float vs int format, but it is safe/easy to just
set to some overly large value such that it won't limit your detection tables,
e.g. `n_det=10000`.
* `sub_tile` at a default of `2` will effectively fit an affine transformation
in each quart of the image. On extremely distorted images even this may not be
enough and so it can be raised to `3` (or even `4`). It is a balancing act 
that there must still be sufficient detections in each image in each sub tile
region from which to make a fit. If you have a low number of detections,
or detections are spread strongly unevenly, this should be set to `1`.
* `spline_order` should generally only be lowered from its default of `3`. 
Setting it to zero might actually be preferable for simple alignment tasks.
Also, with a low number of detections, and particularly with regions of low
number of detections, the splines may misbehave.
* `max_match_dist` is the tolerance in pixels when considering a `source` 
and `template` detection as matched after the affine transform. One may 
increase this in the case of poorly centred detections. Note that this
has an indirect impact on `min_sep` (set to `2 * max_match_dist` by 
default) - when raising `max_match_dist` then `min_sep` correspondingly
increases, offering some guard against ambguous cross-matching in crowded
regions. However, raising it too high may mean that too few detections
pass the `min_sep` criterion. In crowded fields and with well-behaved
detection centres, reducing `max_match_dist` may be advisable.