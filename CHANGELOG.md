# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased
- jinja2 is now correctly a dependency
- Added an optional progress bar to data collection

## v0.1.1 - 2024-01-22

### Added
- Added guided tutorial notebooks
- Added an option to customize the orchestrator and datastore ports
- Added ParallelEnvironment as a default export from envs
- Added a placeholder image for the web UI

### Fixed
- Updated the uvicorn dependency to require the [standard] option
- Fixed a breaking bug in ParallelEnv
- Fixed some type issues, ignore some spurious warnings

### Changed
- Dropped OpenCV as a requirement

## v0.1.0 - 2024-01-17

### Added

- Initial release
