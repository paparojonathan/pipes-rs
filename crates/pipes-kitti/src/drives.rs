//! `list_drives`: what is actually on disk under a KITTI data root.
//!
//! This is dataset knowledge, not CLI knowledge, so it lives beside the driver
//! that consumes it. `--drive` is an unvalidated name; without a
//! listing, the only way to learn what a data root holds was to guess a pair
//! and read the error. Every check here is the one
//! [`crate::cam0::Cam0Driver::open`] and [`crate::velo::VeloDriver::open`]
//! make, run in advance and *reported* rather than raised: a broken drive must
//! still appear in the listing, since seeing the broken one is the reason to
//! ask.

use std::path::Path;

use crate::cam0::{frame_file_name, png_dimensions, png_names};
use crate::layout::FrameIndex;
use crate::timestamps::parse_timestamp_lines;
use crate::velo::VeloDriver;

/// Stand-in `drive` name for a capture-date directory that could not be read
/// at all. Never a real KITTI drive name, and deliberately not an empty string:
/// a blank column in the listing reads as "no problem here".
pub const UNREADABLE_DRIVE: &str = "(unreadable)";

/// One `<root>/<date>/<drive>/image_02` found on disk, and whether it is sound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriveInfo {
    /// Capture-date directory name, e.g. `2011_09_26`.
    pub date: String,
    /// Drive directory name, e.g. `2011_09_26_drive_0005_sync`.
    pub drive: String,
    /// Frames the drive has: the most frame slots any of its sensors
    /// declares -- the lines of `timestamps.txt` or of the lidar's timestamp
    /// files, blank lines included -- as `pipes run` counts them. Each
    /// sensor's frames below it with no sample are absent in its source,
    /// the ones past its own last line included. When the camera's layout
    /// cannot be read, its line count or its PNGs, so the column is never
    /// blank.
    pub frames: usize,
    /// Width of frame 0 in pixels; `0` when its header could not be read.
    /// KITTI resolution varies by capture *date* (1242x375, 1224x370, …), so
    /// this is measured per drive rather than assumed.
    pub width: u32,
    /// Height of frame 0 in pixels; `0` when its header could not be read.
    pub height: u32,
    /// PNGs in `image_02`, as [`crate::cam0::Cam0Driver::open`] reads them:
    /// one per file, one per non-blank line of `timestamps.txt`. `None` when
    /// they break KITTI's layout -- then `problem` says why, because `pipes
    /// run` would refuse the drive for the same reason. `frames` less
    /// `camera_absent`.
    pub images: Option<usize>,
    /// Whether the drive carries a `velodyne_points/` directory.
    ///
    /// Its absence is not a health problem: a camera-only drive is a common
    /// case (many KITTI downloads are `image_02` alone), so this is a fact
    /// about what the drive holds. It is the column `--lidar auto` reads.
    pub lidar: bool,
    /// Sweeps in `velodyne_points`, as [`VeloDriver::open`] reads them: one
    /// per file, one per non-blank line of each timestamp file. `None`
    /// without lidar, or when the lidar could not be opened -- then `problem`
    /// says why, because `pipes run` would refuse the drive for the same
    /// reason.
    pub sweeps: Option<usize>,
    /// Frames of the drive whose sweep is absent in the source, ascending:
    /// `[177, 178, 179, 180]` for drive 0009. Not a `problem` -- the run
    /// replays them as missing inputs -- but a fact about the data a reader
    /// must see before reading its results.
    pub lidar_absent: Vec<u64>,
    /// The same for the camera: frames with no PNG in the source, the ones
    /// past its last line included when the lidar declares more. Empty on
    /// every KITTI drive seen so far.
    pub camera_absent: Vec<u64>,
    /// Whether every check passed, i.e. whether `problem` is empty.
    pub ok: bool,
    /// Every failed check, `; `-joined; empty exactly when `ok`.
    pub problem: String,
}

impl DriveInfo {
    /// `WxH`, or `?` when the first frame's header could not be read.
    pub fn resolution(&self) -> String {
        if self.width == 0 || self.height == 0 {
            "?".to_string()
        } else {
            format!("{}x{}", self.width, self.height)
        }
    }
}

/// Every drive under `root`, sorted by `(date, drive)`.
///
/// A directory is reported when `<root>/<date>/<drive>/image_02` exists; what
/// is wrong with it goes in `problem`, never in the return type. Fails only
/// when `root` itself cannot be listed — the caller's exit-1 case.
///
/// Names that are not valid UTF-8 come back lossily converted, which is good
/// enough to *show* and not good enough to pass back as `--drive`; such a name
/// could not have been typed as an argument anyway.
pub fn list_drives(root: &Path) -> std::io::Result<Vec<DriveInfo>> {
    let mut out: Vec<DriveInfo> = Vec::new();
    let mut dates = read_dir_names(root)?;
    dates.sort();
    for date in dates {
        let date_dir = root.join(&date);
        let drives = match read_dir_names(&date_dir) {
            Ok(mut d) => {
                d.sort();
                d
            }
            // A date directory that exists and cannot be read is not the same
            // as one holding no drives, and silently skipping it would make
            // those two look identical in the listing.
            Err(e) => {
                out.push(DriveInfo {
                    date,
                    drive: UNREADABLE_DRIVE.to_string(),
                    frames: 0,
                    width: 0,
                    height: 0,
                    images: None,
                    lidar: false,
                    sweeps: None,
                    lidar_absent: Vec::new(),
                    camera_absent: Vec::new(),
                    ok: false,
                    problem: format!("cannot list {}: {e}", date_dir.display()),
                });
                continue;
            }
        };
        for drive in drives {
            let image_dir = date_dir.join(&drive).join("image_02");
            if !image_dir.is_dir() {
                continue;
            }
            let lidar = crate::velo::has_velodyne(root, &date, &drive);
            let (mut info, camera) = inspect(date.clone(), drive, &image_dir, lidar);
            let velo = if lidar {
                inspect_lidar(root, &mut info)
            } else {
                None
            };
            // The drive's frame count, as `pipes run` takes it: the most
            // frame slots either sensor declares, each extended to it.
            info.frames = info
                .frames
                .max(velo.as_ref().map_or(0, VeloDriver::frame_slots));
            if let Some(index) = camera {
                info.camera_absent = index.with_slots(info.frames).absent();
            }
            if let Some(v) = velo {
                let v = v.with_frame_slots(info.frames);
                info.sweeps = Some(v.len());
                info.lidar_absent = v.absent_in_source();
            }
            out.push(info);
        }
    }
    Ok(out)
}

/// Opens the drive's lidar exactly as `pipes run` does -- the same
/// [`VeloDriver::open`] -- and hands it back, so its frames can be read
/// against the drive's; when it will not open, reports why as a `problem`
/// rather than raising it, because the run would refuse the drive for the
/// same reason.
fn inspect_lidar(root: &Path, info: &mut DriveInfo) -> Option<VeloDriver> {
    match VeloDriver::open(root, &info.date, &info.drive) {
        Ok(v) => Some(v),
        Err(e) => {
            if !info.problem.is_empty() {
                info.problem.push_str("; ");
            }
            info.problem.push_str(&format!("lidar: {e}"));
            info.ok = false;
            None
        }
    }
}

/// Sorted-later names of the subdirectories of `dir`.
fn read_dir_names(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.path().is_dir() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    Ok(names)
}

/// Runs `open`'s checks against one `image_02` and reports rather than raises:
/// the same [`FrameIndex::read`] `Cam0Driver::open` runs, so the listing and
/// the run cannot disagree about which drive is sound, and a gap in the
/// camera's source is a fact reported here rather than a failure.
///
/// `lidar` is passed in rather than probed here: every check in this function
/// is about `image_02`; the lidar's own are [`inspect_lidar`]'s. The
/// camera's frames are handed back beside the row, so they can be read
/// against the drive's frame count once the lidar's is known.
fn inspect(
    date: String,
    drive: String,
    image_dir: &Path,
    lidar: bool,
) -> (DriveInfo, Option<FrameIndex>) {
    let mut problems: Vec<String> = Vec::new();
    let data = image_dir.join("data");
    let ts_path = image_dir.join("timestamps.txt");

    // The count to fall back on when the layout cannot be read at all.
    let (pngs, listed) = match png_names(&data) {
        Ok(n) => (n.len(), true),
        Err(e) => {
            problems.push(format!("cannot list {}: {e}", data.display()));
            (0, false)
        }
    };
    // `(frames, the first PNG's frame, the camera's frames)`.
    let (frames, first, index) = match parse_timestamp_lines(&ts_path) {
        Ok(ts) => match FrameIndex::read(&data, "png", &ts_path, &ts) {
            Ok(index) => (index.slots(), index.frame(0), Some(index)),
            Err(e) => {
                // An unreadable directory is listed once, above, not twice.
                if listed {
                    problems.push(e.to_string());
                }
                (ts.len(), Some(0), None)
            }
        },
        Err(e) => {
            problems.push(e.to_string());
            (pngs, Some(0), None)
        }
    };

    let (width, height) = match (frames, first) {
        (0, _) | (_, None) => {
            problems.push("no frames".to_string());
            (0, 0)
        }
        (_, Some(f)) => {
            let name = frame_file_name(usize::try_from(f).unwrap_or(usize::MAX));
            match png_dimensions(&data.join(&name)) {
                Ok(d) => d,
                Err(e) => {
                    problems.push(format!("data/{name}: {e}"));
                    (0, 0)
                }
            }
        }
    };

    let info = DriveInfo {
        date,
        drive,
        frames,
        width,
        height,
        images: index.as_ref().map(FrameIndex::len),
        lidar,
        sweeps: None,
        lidar_absent: Vec::new(),
        camera_absent: index.as_ref().map(FrameIndex::absent).unwrap_or_default(),
        ok: problems.is_empty(),
        problem: problems.join("; "),
    };
    (info, index)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn resolution_is_a_question_mark_when_unmeasured() {
        let mut d = DriveInfo {
            date: "d".to_string(),
            drive: "v".to_string(),
            frames: 1,
            width: 1242,
            height: 375,
            images: Some(154),
            lidar: false,
            sweeps: None,
            lidar_absent: Vec::new(),
            camera_absent: Vec::new(),
            ok: true,
            problem: String::new(),
        };
        assert_eq!(d.resolution(), "1242x375");
        d.width = 0;
        assert_eq!(d.resolution(), "?");
        d.width = 1242;
        d.height = 0;
        assert_eq!(d.resolution(), "?");
    }

    /// `lidar` is a fact about the drive, not a health check: a camera-only
    /// drive is `ok`. Reported the other way round it would mark every
    /// `image_02`-only download broken.
    #[test]
    fn a_drive_without_lidar_is_listed_and_healthy() {
        use crate::testing::{write_fixture_at, write_velo_fixture_at, FIXTURE_DATE};
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        for drive in ["drive_cam_only", "drive_with_lidar"] {
            write_fixture_at(root, FIXTURE_DATE, drive, 4, 100_000_000).unwrap();
        }
        write_velo_fixture_at(root, FIXTURE_DATE, "drive_with_lidar", 4, 100_000_000).unwrap();

        let found = list_drives(root).unwrap();
        let of = |name: &str| {
            found
                .iter()
                .find(|d| d.drive == name)
                .unwrap_or_else(|| panic!("{name} is missing from {found:?}"))
        };
        // Both listed, both healthy, and they differ in exactly one column --
        // which is what makes the `false` a reading rather than a default.
        assert!(!of("drive_cam_only").lidar);
        assert!(of("drive_with_lidar").lidar);
        for name in ["drive_cam_only", "drive_with_lidar"] {
            assert!(of(name).ok, "{name}: {}", of(name).problem);
            assert_eq!(of(name).frames, 4);
        }
    }
}
