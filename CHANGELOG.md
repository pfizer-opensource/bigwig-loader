# BigWig Loader

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]
...

## [v0.2.0]
### Fixed
- fixed a bug in the intervals to values cuda kernel that
  introduced zeros in places where there should be
  "default_value" (see release v0.1.5).
### Added
- custom_position_sampler argument to bigwig_loader.dataset.BigWigDataset
  and bigwig_loader.pytorch.PytorchBigWigDataset to optionally overwrite the
  default random sampling of genomic coordinates from "regions of interest."
- custom_track_sampler argument to bigwig_loader.dataset.BigWigDataset
  and bigwig_loader.pytorch.PytorchBigWigDataset to optionally use a different
  track sampling strategy.

## [v0.1.5]
### Added
- set a default value different from 0.0

## [v0.1.4]
### Fixed
- Updated README.md with better install instructions. Making release so
  that the README.md is updated on pypi.

## [v0.1.3]
### Fixed
- MANIFEST.in was still not excluding things correctly.

## [v0.1.2]
### Fixed
- exclude .egg-info from source dist

## [v0.1.1]
### Fixed
- fixed urls field in pyproject.toml

## [v0.1.0]
### Fixed
- c files were accidentally pushed to source dist. Now
  they are excluded.

## [v0.0.3]
### Fixed
- remove --no-deps in build process on CI

## [v0.0.2]
### Fixed
- fixed bug in the CI

## [v0.0.1] - 2024-09-20
### Added
- release to pypi

[Unreleased]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.2.0...HEAD
[v0.1.6]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.1.5...v0.2.0
[v0.1.5]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.1.4...v0.1.5
[v0.1.4]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.1.3...v0.1.4
[v0.1.3]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.0.3...v0.1.0
[v0.0.3]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.0.2...v0.0.3
[v0.0.2]: https://github.com/pfizer-opensource/bigwig-loader/compare/v0.0.1...v0.0.2
[v0.0.1]: https://github.com/pfizer-opensource/bigwig-loader/tree/v0.0.1
