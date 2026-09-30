//! `list_drives` against synthetic data roots.
//!
//! The listing exists so a user can see what is on disk *including the parts
//! that are broken*, so almost every test here is a pair: the broken drive is
//! reported as broken, and a sound drive sitting next to it in the same root is
//! still reported as sound. A listing that failed, or that hid the bad entry,
//! would satisfy half of each pair and none of the point.
//!
//! This is its own crate root, so it inherits none of the library's
//! `#![deny(clippy::unwrap_used)]`.

use pipes_kitti::drives::{list_drives, DriveInfo, UNREADABLE_DRIVE};
use pipes_kitti::testing::{
    write_fixture_at, write_fixture_gapped_at, write_velo_fixture_gapped_at, FixtureDrive,
    FIXTURE_DATE, FIXTURE_DRIVE, FIXTURE_H, FIXTURE_PERIOD_NS, FIXTURE_W,
};
use tempfile::TempDir;

/// A second drive, on a second capture date, so the walk is tested two levels
/// deep rather than "the one directory that happens to be there".
const OTHER_DATE: &str = "2011_09_30";
const OTHER_DRIVE: &str = "2011_09_30_drive_0028_sync";

fn find<'a>(drives: &'a [DriveInfo], date: &str, drive: &str) -> &'a DriveInfo {
    drives
        .iter()
        .find(|d| d.date == date && d.drive == drive)
        .unwrap_or_else(|| {
            panic!(
                "{date}/{drive} is not in the listing: {:?}",
                drives
                    .iter()
                    .map(|d| format!("{}/{}", d.date, d.drive))
                    .collect::<Vec<_>>()
            )
        })
}

#[test]
fn lists_one_sound_drive_with_its_frames_and_resolution() {
    let fx = FixtureDrive::new(16).unwrap();
    let drives = list_drives(fx.root()).unwrap();
    assert_eq!(drives.len(), 1, "{drives:?}");
    let d = &drives[0];
    assert_eq!(d.date, FIXTURE_DATE);
    assert_eq!(d.drive, FIXTURE_DRIVE);
    assert_eq!(d.frames, 16);
    // Read from the PNG header, not assumed: KITTI's resolution differs by
    // capture date, so a hard-coded 1242x375 would be wrong four dates out of
    // five.
    assert_eq!((d.width, d.height), (FIXTURE_W, FIXTURE_H));
    assert_eq!(d.resolution(), format!("{FIXTURE_W}x{FIXTURE_H}"));
    assert!(d.ok, "problem: {}", d.problem);
    assert_eq!(d.problem, "");
}

#[test]
fn walks_two_levels_and_sorts_by_date_then_drive() {
    let tmp = TempDir::new().unwrap();
    // Written out of order, so a listing that merely echoed directory order
    // would come back out of order too on at least one filesystem.
    write_fixture_at(tmp.path(), OTHER_DATE, OTHER_DRIVE, 4, FIXTURE_PERIOD_NS).unwrap();
    write_fixture_at(
        tmp.path(),
        FIXTURE_DATE,
        "2011_09_26_drive_0009_sync",
        3,
        FIXTURE_PERIOD_NS,
    )
    .unwrap();
    write_fixture_at(
        tmp.path(),
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        2,
        FIXTURE_PERIOD_NS,
    )
    .unwrap();

    let drives = list_drives(tmp.path()).unwrap();
    let keys: Vec<String> = drives
        .iter()
        .map(|d| format!("{}/{}", d.date, d.drive))
        .collect();
    assert_eq!(
        keys,
        vec![
            format!("{FIXTURE_DATE}/{FIXTURE_DRIVE}"),
            format!("{FIXTURE_DATE}/2011_09_26_drive_0009_sync"),
            format!("{OTHER_DATE}/{OTHER_DRIVE}"),
        ]
    );
    assert_eq!(drives[0].frames, 2);
    assert_eq!(drives[1].frames, 3);
    assert_eq!(drives[2].frames, 4);
    assert!(drives.iter().all(|d| d.ok), "{drives:?}");
}

#[test]
fn a_broken_drive_is_listed_beside_a_sound_one() {
    let tmp = TempDir::new().unwrap();
    write_fixture_at(
        tmp.path(),
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        8,
        FIXTURE_PERIOD_NS,
    )
    .unwrap();
    write_fixture_at(tmp.path(), OTHER_DATE, OTHER_DRIVE, 8, FIXTURE_PERIOD_NS).unwrap();

    // The P4 shape again: right count, wrong name.
    let data = tmp
        .path()
        .join(OTHER_DATE)
        .join(OTHER_DRIVE)
        .join("image_02")
        .join("data");
    std::fs::rename(data.join("0000000005.png"), data.join("oops.png")).unwrap();

    let drives = list_drives(tmp.path()).unwrap();
    assert_eq!(
        drives.len(),
        2,
        "a broken drive must not vanish: {drives:?}"
    );

    let broken = find(&drives, OTHER_DATE, OTHER_DRIVE);
    assert!(!broken.ok, "the renamed frame was not noticed");
    // Named by the file that is not a frame: which frame it was meant to be
    // is exactly what cannot be known without guessing.
    assert!(
        broken.problem.contains("oops.png"),
        "the problem must name the file that is not a frame: {}",
        broken.problem
    );
    assert_eq!(broken.frames, 8, "frames still comes from timestamps.txt");

    // Positive control: the drive that was not touched is still sound, so
    // `ok == false` above means something.
    let sound = find(&drives, FIXTURE_DATE, FIXTURE_DRIVE);
    assert!(sound.ok, "problem: {}", sound.problem);
    assert_eq!(sound.problem, "");
}

#[test]
fn a_drive_with_no_timestamps_is_listed_as_broken() {
    let fx = FixtureDrive::new(8).unwrap();
    std::fs::remove_file(fx.image_dir().join("timestamps.txt")).unwrap();
    let drives = list_drives(fx.root()).unwrap();
    assert_eq!(drives.len(), 1);
    let d = &drives[0];
    assert!(!d.ok, "a drive with no timestamps.txt is not sound");
    assert!(
        d.problem.contains("timestamps.txt"),
        "problem: {}",
        d.problem
    );
    // The count falls back to the PNGs rather than leaving the column blank,
    // and the resolution is still measured, because both are still knowable.
    assert_eq!(d.frames, 8);
    assert_eq!((d.width, d.height), (FIXTURE_W, FIXTURE_H));
}

#[test]
fn a_drive_whose_first_frame_is_not_a_png_reports_a_question_mark() {
    let fx = FixtureDrive::new(4).unwrap();
    std::fs::write(fx.frame_path(0), b"this is not a PNG").unwrap();
    let drives = list_drives(fx.root()).unwrap();
    let d = &drives[0];
    assert!(!d.ok);
    assert_eq!((d.width, d.height), (0, 0));
    assert_eq!(d.resolution(), "?");
    assert!(
        d.problem.contains("0000000000.png"),
        "problem: {}",
        d.problem
    );
    // Contiguity still passed: the file is there, it is simply not an image.
    assert_eq!(d.frames, 4);
}

#[test]
fn a_directory_without_image_02_is_not_a_drive() {
    let fx = FixtureDrive::new(2).unwrap();
    // KITTI's other sensor directories, and the calib files that sit beside
    // the drives, must not turn into rows.
    std::fs::create_dir_all(fx.root().join(FIXTURE_DATE).join("not_a_drive")).unwrap();
    std::fs::create_dir_all(
        fx.root()
            .join(FIXTURE_DATE)
            .join("velodyne_only")
            .join("velodyne_points"),
    )
    .unwrap();
    std::fs::write(fx.root().join("MANIFEST.txt"), "").unwrap();
    std::fs::write(
        fx.root().join(FIXTURE_DATE).join("calib_cam_to_cam.txt"),
        "",
    )
    .unwrap();

    let drives = list_drives(fx.root()).unwrap();
    assert_eq!(drives.len(), 1, "{drives:?}");
    assert_eq!(drives[0].drive, FIXTURE_DRIVE);
}

#[test]
fn an_empty_root_lists_nothing_and_is_not_an_error() {
    let tmp = TempDir::new().unwrap();
    let drives = list_drives(tmp.path()).unwrap();
    assert!(drives.is_empty(), "{drives:?}");

    // A date directory holding no drives at all is the same answer.
    std::fs::create_dir_all(tmp.path().join(FIXTURE_DATE)).unwrap();
    assert!(list_drives(tmp.path()).unwrap().is_empty());
}

#[test]
fn a_missing_root_is_an_error() {
    let tmp = TempDir::new().unwrap();
    let absent = tmp.path().join("no_such_root");
    let Err(e) = list_drives(&absent) else {
        panic!("list_drives accepted a root that does not exist");
    };
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "{e}");

    // Positive control: the *same* call against a root that does exist
    // succeeds, so the Err above is about the path and not about the walk.
    assert!(list_drives(tmp.path()).is_ok());
}

/// A file where a capture-date directory is expected is skipped, not fatal;
/// `UNREADABLE_DRIVE` is reserved for a date directory that exists and cannot
/// be listed, which no portable test can stage, so the constant is pinned here
/// instead of being left to drift into an empty string.
#[test]
fn the_unreadable_marker_is_not_mistakable_for_a_drive() {
    assert!(!UNREADABLE_DRIVE.is_empty());
    assert!(!UNREADABLE_DRIVE.contains("drive_"));
}

/// The listing must agree with `Cam0Driver::open`: a drive the listing calls
/// sound must open, and one it calls broken must not. Two independent
/// implementations of "is this drive usable" that disagree would be worse than
/// having only one.
#[test]
fn listing_health_agrees_with_driver_open() {
    let fx = FixtureDrive::new(8).unwrap();
    let sound = &list_drives(fx.root()).unwrap()[0];
    assert!(sound.ok);
    assert!(pipes_kitti::cam0::Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).is_ok());

    fx.rename_in_data("0000000003.png", "three.png").unwrap();
    let broken = &list_drives(fx.root()).unwrap()[0];
    assert!(!broken.ok, "problem: {:?}", broken.problem);
    assert!(pipes_kitti::cam0::Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).is_err());
}

/// A gap in a sensor's source is a fact the listing reports, not a failure:
/// the drive is `ok`, its sweeps are counted the way `run` counts them, and
/// the frames with no sample are named -- for the lidar and the camera alike.
#[test]
fn a_gap_in_either_source_is_listed_as_a_fact_and_the_drive_is_sound() {
    let tmp = TempDir::new().unwrap();
    // The lidar lacks frames 3-4; the other drive's camera lacks frame 6.
    write_fixture_at(
        tmp.path(),
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        10,
        FIXTURE_PERIOD_NS,
    )
    .unwrap();
    write_velo_fixture_gapped_at(
        tmp.path(),
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        10,
        &[3, 4],
        FIXTURE_PERIOD_NS,
    )
    .unwrap();
    write_fixture_gapped_at(
        tmp.path(),
        OTHER_DATE,
        OTHER_DRIVE,
        10,
        &[6],
        FIXTURE_PERIOD_NS,
    )
    .unwrap();

    let drives = list_drives(tmp.path()).unwrap();
    let lidar = find(&drives, FIXTURE_DATE, FIXTURE_DRIVE);
    assert!(
        lidar.ok,
        "a gap in the source is not a broken drive: {}",
        lidar.problem
    );
    assert_eq!(
        (lidar.frames, lidar.images, lidar.sweeps),
        (10, Some(10), Some(8))
    );
    assert_eq!(lidar.lidar_absent, vec![3, 4]);
    assert!(lidar.camera_absent.is_empty());

    let camera = find(&drives, OTHER_DATE, OTHER_DRIVE);
    assert!(camera.ok, "{}", camera.problem);
    assert_eq!(camera.frames, 10, "frame slots, the absent one included");
    assert_eq!(camera.images, Some(9), "PNGs on disk");
    assert_eq!(camera.camera_absent, vec![6]);
    assert!(!camera.lidar);
    assert_eq!(camera.sweeps, None);
    // The resolution is still measured, off the first PNG there is.
    assert_eq!((camera.width, camera.height), (FIXTURE_W, FIXTURE_H));
}

/// A lidar that stops before the camera, and one whose timestamp files
/// declare its last frames with blank lines: either way the drive has ten
/// frames and the lidar's last two are absent in its source. In the first,
/// only the camera's frame count can say so; the lidar's own files cannot.
#[test]
fn a_lidar_that_stops_early_is_listed_with_its_last_frames_absent() {
    let fx = FixtureDrive::with_velodyne(10, 8, FIXTURE_PERIOD_NS).unwrap();
    let d = &list_drives(fx.root()).unwrap()[0];
    assert!(d.ok, "{}", d.problem);
    assert_eq!((d.frames, d.images, d.sweeps), (10, Some(10), Some(8)));
    assert_eq!(d.lidar_absent, vec![8, 9]);
    assert!(d.camera_absent.is_empty());

    let fx = FixtureDrive::with_gaps(10, &[], &[8, 9], FIXTURE_PERIOD_NS).unwrap();
    let d = &list_drives(fx.root()).unwrap()[0];
    assert!(d.ok, "{}", d.problem);
    assert_eq!((d.frames, d.images, d.sweeps), (10, Some(10), Some(8)));
    assert_eq!(d.lidar_absent, vec![8, 9]);
}

/// A camera that stops before the lidar, and one whose `timestamps.txt`
/// declares its last frames with blank lines: either way the drive has ten
/// frames and the camera's last two are absent in its source.
#[test]
fn a_camera_that_stops_early_is_listed_with_its_last_frames_absent() {
    let fx = FixtureDrive::with_velodyne(8, 10, FIXTURE_PERIOD_NS).unwrap();
    let d = &list_drives(fx.root()).unwrap()[0];
    assert!(d.ok, "{}", d.problem);
    assert_eq!((d.frames, d.images, d.sweeps), (10, Some(8), Some(10)));
    assert_eq!(d.camera_absent, vec![8, 9]);
    assert!(d.lidar_absent.is_empty());

    let fx = FixtureDrive::with_gaps(10, &[8, 9], &[], FIXTURE_PERIOD_NS).unwrap();
    let d = &list_drives(fx.root()).unwrap()[0];
    assert!(d.ok, "{}", d.problem);
    assert_eq!((d.frames, d.images, d.sweeps), (10, Some(8), Some(10)));
    assert_eq!(d.camera_absent, vec![8, 9]);
}

/// A lidar whose files and lines break KITTI's layout is a
/// broken drive -- `run` would refuse it for the same reason -- and the
/// listing says so, with the lidar's own message.
#[test]
fn a_lidar_that_would_not_open_is_listed_as_broken() {
    let fx = FixtureDrive::with_velodyne(8, 8, FIXTURE_PERIOD_NS).unwrap();
    let sound = &list_drives(fx.root()).unwrap()[0];
    assert!(sound.ok, "positive control: {}", sound.problem);
    assert_eq!(sound.sweeps, Some(8));
    // A sweep file deleted and its lines kept: 7 files against 8 lines.
    std::fs::remove_file(fx.velo_dir().join("data").join("0000000005.bin")).unwrap();
    let broken = &list_drives(fx.root()).unwrap()[0];
    assert!(!broken.ok);
    assert!(
        broken.problem.starts_with("lidar: ") && broken.problem.contains("would be a guess"),
        "{}",
        broken.problem
    );
    assert_eq!(
        broken.sweeps, None,
        "a count the run could not use is not reported"
    );
    // The listing and the driver agree.
    assert!(pipes_kitti::velo::VeloDriver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).is_err());
}
