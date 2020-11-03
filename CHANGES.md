# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2020-11

 - Changelog added!
 - Internal detection routines added using [`sep`](https://github.com/kbarbary/sep).
 - `align-fits` and `align-fits-simple` console scripts added.
 - Allow initial affine transformation cross-matching to be performed in `sub_tiles`
 - Check multiple quads and find best cross-match performing quad, instead of blindly accepting the first.
 - Lots of refactoring and backwards-breaking argument changes.