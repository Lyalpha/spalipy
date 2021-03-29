# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2021-03
 - Can now align multiple images in sequence by passing lists for arguments relating to
   source data, detections, shapes etc.
 - Can now pass a `source_mask` to the main `Spalipy` constructor. This mask will be transformed, using
   nearest neighbour interpolation, in the same manner as the `source_data`.
   - Note this functionality is not yet possible from the cli (i.e. `align-fits`)
 - Changes to `Spalipy` calling signature:
   - Make `n_det` optional. Default `None` will now not cut the length of the detections
   - Reduce default `sub_tile` from `2` to `1`
   - Reduce default `max_match_dist` from `5` to `3`
 - Fixed bug when calling cli scripts without full arguments specified
 - Change `sep` version requirements from `1.0.3` to `1.1.1`
 - First appearance of actual tests(!)

## [2.0.3] - 2020-11

 - Fix bug on parameter names in console scripts
 
## [2.0.2] - 2020-11

 - Changelog added!
 - Internal detection routines added using [`sep`](https://github.com/kbarbary/sep).
 - `align-fits` and `align-fits-simple` console scripts added.
 - Allow initial affine transformation cross-matching to be performed in `sub_tiles`
 - Check multiple quads and find best cross-match performing quad, instead of blindly accepting the first.
 - Lots of refactoring and backwards-breaking argument changes.