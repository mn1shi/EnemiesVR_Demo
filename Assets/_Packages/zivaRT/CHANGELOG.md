# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2022-03-17
### Added
- Add a property to the ZivaRTPlayer component to control solver scheduling. Solver can be started manually 
  or based on visibility in either "Update" or "LateUpdate"
- Add functionality to specify custom bounds for ZivaRTPlayer component 

### Changed
- Ensure that target mesh is in the format expected by the "Compute Shader" solver
- Shader data is now stored as a package resource instead of being serialized with ZivaRTPlayer
- Improve "Burst Jobs" solver performance by using Mesh.GetVertexBuffer API to update target mesh data 
- Refactored source mesh change detection to use the custom inspector
- Update the documentation of the "Draw Debug Vertices" field to better describe what it does
- Make sure all public ZivaRT player's fields have tooltips

### Fixed
- Fix motion vectors having uninitialized values the first time a solver is run
- Make 2022.2 the earliest version that supports the compute shader solver on Apple devices
- Fix occasional crash and instability in AnimateExtraParametersTests
- Fix instability in Runtime Playable Tests
- Fix Editor errors when linked mesh is missing on imported ZivaRTPlayer component
- Fix bounds not being calculated when using "Compute" solver
- Fix target mesh not being created when the character's pivot point is outside of camera's view

### Removed
- Removed some unneeded files getting included with the package

## [0.1.0] - 2023-01-13
This is the initial release of ZivaRT Player.
