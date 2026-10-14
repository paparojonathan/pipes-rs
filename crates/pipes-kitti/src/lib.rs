#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]
//! KITTI dataset drivers: the timestamps parser, the PNG cam0 driver and its
//! Arrow frame builder, the velodyne lidar driver and its Arrow sweep
//! builder, and the OXTS GPS/IMU driver and its Arrow row builder.

pub mod calib; // M14: the lidar -> camera projection, and the files it is parsed from
pub mod cam0; // M1: Cam0Driver
pub mod camdet; // the frozen camera detector: YOLOX-Nano on each frame, emitted as CAM_DET
pub mod detect; // M13: ground removal and clustering — the first stage whose output answers something
pub mod driver; // DriverEvent/RunTotals and the checks every driver makes
pub mod drives; // list_drives: what a data root actually holds
pub mod frame; // M0: cam0_schema, build_cam0_batch, cam0_pixels; tests land in M2
pub mod fuse; // the association: which camera detection is which lidar track, and the three populations
pub mod layout; // which frames a sensor directory holds, gaps in the source included, for both drivers
pub mod name; // which lidar-detector box is which lidar track: the class a track takes from the lidar
pub mod onnx; // ONNX Runtime, loaded once and a session built the same way for either detector
pub mod oxts; // OxtsDriver, the third stream: the vehicle's own motion, for the fusion to subtract
pub mod path; // the arc the vehicle is on, from its speed and yaw rate, and what stands on it
pub mod pillars; // the frozen lidar detector: PointPillars on each sweep, emitted as PILLAR_DET
pub mod state; // M14: the answer -- every tracked object, the nearest in the vehicle's path flagged
pub mod timestamps; // M1: parse_timestamps
pub mod track; // M14: detections associated across sweeps, fused with the camera frame
pub mod velo; // M11: VeloDriver, the second stream
pub mod voxel; // M12: the voxel reduction — the first stage that produces Arrow

/// Synthetic KITTI drive for tests; compiled out of normal builds.
///
/// `cfg(test)` alone would not do: that is visible only to this crate's own
/// unit tests, not to its integration tests and not to `crates/pipes`, which
/// needs the fixture to drive the real binary.
#[cfg(any(test, feature = "testing"))]
pub mod testing;
