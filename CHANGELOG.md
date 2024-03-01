# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

## v0.1.4 - 2024-03-02

## v0.1.4 - 2024-02-09

### Changed
- Web UI now receives both the observation and the rendered frame
- Web UI can work on http and https

## v0.1.3 - 2024-02-09

### Fixed
- `cogmentlab install` no longer requires root access

### Changed
- The `cogmentlab` command now uses the `cogment` executable stored in `$COGMENT_LAB_HOME/cogment`. This might require rerunning `cogmentlab install` or updating the environment variable.


## v0.1.2 - 2024-02-02

### Added
- Added an optional progress bar to data collection, with tqdm as a new dependency

### Fixed
- jinja2 is now correctly a dependency


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
