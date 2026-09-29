//! `pipes drives`: print every drive under the data root.
//!
//! Thin on purpose — the walk and every health check live in
//! `pipes_kitti::drives`, because they are facts about KITTI rather than about
//! this binary. What is here is the two output shapes and the exit code.

use std::path::Path;

use pipes_kitti::drives::{list_drives, DriveInfo};
use pipes_kitti::layout::frames_phrase;
use serde::Serialize;

use crate::cli::DrivesArgs;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// One element of the `--json` array. The key order is the contract, so it is
/// a struct with named fields rather than an ad-hoc map.
#[derive(Serialize)]
struct DriveJson<'a> {
    date: &'a str,
    drive: &'a str,
    frames: usize,
    /// `0` when frame 0's header could not be read; `problem` says why.
    width: u32,
    /// `0` when frame 0's header could not be read; `problem` says why.
    height: u32,
    /// PNGs in `image_02`, paired one to one with the lines of
    /// `timestamps.txt`: `frames` less `camera_absent_in_source`. `null` when
    /// they could not be paired (then `problem` says why).
    images: Option<usize>,
    /// Whether the drive carries a `velodyne_points/` directory, i.e. whether
    /// `pipes run --lidar auto` replays a second stream from it.
    ///
    /// Its absence is not a health problem: a camera-only drive is a fact
    /// about what the drive holds, and never a `problem`.
    lidar: bool,
    /// Sweeps in `velodyne_points`: `null` without lidar, or when it could
    /// not be opened (then `problem` says why).
    sweeps: Option<usize>,
    /// Frames whose sweep is absent in the source, ascending; empty when the
    /// lidar has a sweep for every frame. A property of the data, not a
    /// `problem`: `run` replays them as missing inputs.
    lidar_absent_in_source: &'a [u64],
    /// The same for the camera: frames with no PNG in the source.
    camera_absent_in_source: &'a [u64],
    ok: bool,
    /// Empty exactly when `ok`, the same way an empty `problem` is spelled in
    /// the table.
    problem: &'a str,
}

/// Exit 0 whenever the root could be listed — including when it holds no
/// drives at all, which is an answer and not a failure. Exit 1 (through
/// `main`) only when the root itself is missing or unreadable.
pub fn main(a: DrivesArgs) -> Result<(), BoxError> {
    let root = &a.kitti_root;
    let drives = list_drives(root).map_err(|e| -> BoxError {
        format!(
            "cannot read the KITTI data root {}{}: {e}. Pass --kitti-root <dir>, \
             or set PIPES_KITTI_ROOT once; the default expects the dataset \
             beside the checkout",
            root.display(),
            cwd_hint(root)
        )
        .into()
    })?;
    if a.json {
        print_json(&drives)?;
    } else {
        print_table(root, &drives);
    }
    Ok(())
}

fn print_json(drives: &[DriveInfo]) -> Result<(), BoxError> {
    let rows: Vec<DriveJson<'_>> = drives
        .iter()
        .map(|d| DriveJson {
            date: &d.date,
            drive: &d.drive,
            frames: d.frames,
            width: d.width,
            height: d.height,
            images: d.images,
            lidar: d.lidar,
            sweeps: d.sweeps,
            lidar_absent_in_source: &d.lidar_absent,
            camera_absent_in_source: &d.camera_absent,
            ok: d.ok,
            problem: &d.problem,
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

/// The `camera` column: PNGs against the drive's frames (`447/447`), `?`
/// when they could not be paired with the timestamps (the health column says
/// why).
fn camera_cell(d: &DriveInfo) -> String {
    match d.images {
        Some(n) => format!("{n}/{}", d.frames),
        None => "?".to_string(),
    }
}

/// The `lidar` column: sweeps against the drive's frames (`443/447`), `-`
/// for a drive with no `velodyne_points/`, `?` for one whose lidar would not
/// open (the health column says why).
fn lidar_cell(d: &DriveInfo) -> String {
    match (d.lidar, d.sweeps) {
        (false, _) => "-".to_string(),
        (true, Some(n)) => format!("{n}/{}", d.frames),
        (true, None) => "?".to_string(),
    }
}

/// What the health column adds after `ok`: which frames either sensor's
/// source has no sample for, and that a drive is camera-only. Empty
/// otherwise.
fn health_note(d: &DriveInfo) -> String {
    let mut notes: Vec<String> = Vec::new();
    if !d.camera_absent.is_empty() {
        notes.push(format!(
            "camera gap: {} absent in source",
            frames_phrase(&d.camera_absent)
        ));
    }
    if !d.lidar {
        notes.push("camera-only".to_string());
    } else if !d.lidar_absent.is_empty() {
        notes.push(format!(
            "lidar gap: {} absent in source",
            frames_phrase(&d.lidar_absent)
        ));
    }
    if notes.is_empty() {
        String::new()
    } else {
        format!(" ({})", notes.join("; "))
    }
}

/// Columns sized to the data, so a long drive name never shifts the numbers
/// out from under the header.
fn print_table(root: &Path, drives: &[DriveInfo]) {
    if drives.is_empty() {
        println!("no drives found under {}{}", root.display(), cwd_hint(root));
        println!(
            "a drive is <root>/<date>/<drive>/image_02, e.g. \
             {}/2011_09_26/2011_09_26_drive_0005_sync/image_02",
            root.display()
        );
        return;
    }
    let resolutions: Vec<String> = drives.iter().map(DriveInfo::resolution).collect();
    let cameras: Vec<String> = drives.iter().map(camera_cell).collect();
    let lidars: Vec<String> = drives.iter().map(lidar_cell).collect();
    let w_date = width("date", drives.iter().map(|d| d.date.as_str()));
    let w_drive = width("drive", drives.iter().map(|d| d.drive.as_str()));
    let w_frames = width("frames", drives.iter().map(|d| d.frames.to_string()));
    let w_res = width("resolution", resolutions.iter().map(String::as_str));
    let w_camera = width("camera", cameras.iter().map(String::as_str));
    let w_lidar = width("lidar", lidars.iter().map(String::as_str));

    println!(
        "{:<w_date$}  {:<w_drive$}  {:>w_frames$}  {:<w_res$}  {:<w_camera$}  {:<w_lidar$}  health",
        "date", "drive", "frames", "resolution", "camera", "lidar"
    );
    for (((d, res), camera), lidar) in drives.iter().zip(&resolutions).zip(&cameras).zip(&lidars) {
        let health = if d.ok {
            format!("ok{}", health_note(d))
        } else {
            format!("BROKEN  {}", d.problem)
        };
        println!(
            "{:<w_date$}  {:<w_drive$}  {:>w_frames$}  {:<w_res$}  {camera:<w_camera$}  {lidar:<w_lidar$}  {health}",
            d.date, d.drive, d.frames, res
        );
    }
    let broken = drives.iter().filter(|d| !d.ok).count();
    let gapped = drives
        .iter()
        .filter(|d| !d.lidar_absent.is_empty() || !d.camera_absent.is_empty())
        .count();
    println!(
        "{} drive(s) under {}, {broken} broken{}",
        drives.len(),
        root.display(),
        if gapped > 0 {
            format!(", {gapped} with a gap in the source")
        } else {
            String::new()
        }
    );
}

/// Widest of the header and the column's values.
fn width<S: AsRef<str>>(header: &str, values: impl Iterator<Item = S>) -> usize {
    values
        .map(|v| v.as_ref().chars().count())
        .chain(std::iter::once(header.chars().count()))
        .max()
        .unwrap_or(0)
}

/// ` (relative to <cwd>)`, but only when the root is relative and so ambiguous
/// on its own — the same hint `DriverError::DriveMissing` gives.
fn cwd_hint(root: &Path) -> String {
    if !root.is_relative() {
        return String::new();
    }
    match std::env::current_dir() {
        Ok(cwd) => format!(" (relative to {})", cwd.display()),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn column_width_is_the_widest_of_header_and_values() {
        assert_eq!(width("frames", ["1", "154"].into_iter()), 6);
        assert_eq!(width("date", ["2011_09_26"].into_iter()), 10);
        // No values at all still leaves room for the header, or the table
        // would print a header wider than its own separator.
        assert_eq!(width("resolution", std::iter::empty::<&str>()), 10);
    }

    fn info(lidar: bool, sweeps: Option<usize>, absent: Vec<u64>) -> DriveInfo {
        DriveInfo {
            date: "2011_09_26".to_string(),
            drive: "2011_09_26_drive_0009_sync".to_string(),
            frames: 447,
            width: 1242,
            height: 375,
            images: Some(447),
            lidar,
            sweeps,
            lidar_absent: absent,
            camera_absent: Vec::new(),
            ok: true,
            problem: String::new(),
        }
    }

    #[test]
    fn json_keys_are_the_documented_twelve() {
        let d = info(true, Some(443), vec![177, 178, 179, 180]);
        let row = DriveJson {
            date: &d.date,
            drive: &d.drive,
            frames: d.frames,
            width: d.width,
            height: d.height,
            images: d.images,
            lidar: d.lidar,
            sweeps: d.sweeps,
            lidar_absent_in_source: &d.lidar_absent,
            camera_absent_in_source: &d.camera_absent,
            ok: d.ok,
            problem: &d.problem,
        };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&row).unwrap())
            .expect("DriveJson serialises to JSON");
        let obj = v.as_object().expect("an object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "camera_absent_in_source",
                "date",
                "drive",
                "frames",
                "height",
                "images",
                "lidar",
                "lidar_absent_in_source",
                "ok",
                "problem",
                "sweeps",
                "width"
            ]
        );
        assert_eq!(obj["ok"], serde_json::json!(true));
        assert_eq!(obj["problem"], serde_json::json!(""));
        assert_eq!(obj["frames"], serde_json::json!(447));
        // A bool, not a string: a consumer of `--json` filters on it.
        assert_eq!(obj["lidar"], serde_json::json!(true));
        assert_eq!(obj["sweeps"], serde_json::json!(443));
        assert_eq!(obj["images"], serde_json::json!(447));
        assert_eq!(
            obj["lidar_absent_in_source"],
            serde_json::json!([177, 178, 179, 180])
        );
    }

    /// The cells a reader reads a drive's data off: each sensor's samples
    /// against the drive's frames, and a gap by name -- or `-` and
    /// `camera-only`, so a drive with no lidar reads as a fact rather than a
    /// failure.
    #[test]
    fn the_cells_say_samples_against_frames_and_name_the_gap() {
        let gapped = info(true, Some(443), vec![177, 178, 179, 180]);
        assert_eq!(camera_cell(&gapped), "447/447");
        assert_eq!(lidar_cell(&gapped), "443/447");
        assert_eq!(
            health_note(&gapped),
            " (lidar gap: frames 177-180 absent in source)"
        );
        let whole = DriveInfo {
            frames: 154,
            images: Some(154),
            ..info(true, Some(154), Vec::new())
        };
        assert_eq!(lidar_cell(&whole), "154/154");
        assert_eq!(health_note(&whole), "");
        let camera_only = info(false, None, Vec::new());
        assert_eq!(lidar_cell(&camera_only), "-");
        assert_eq!(health_note(&camera_only), " (camera-only)");
        // A lidar that would not open: the count is unknown, not zero.
        assert_eq!(lidar_cell(&info(true, None, Vec::new())), "?");
        // A gap in the camera's source is named too, beside the lidar's.
        let both = DriveInfo {
            images: Some(446),
            camera_absent: vec![7],
            ..info(true, Some(443), vec![177, 178, 179, 180])
        };
        assert_eq!(camera_cell(&both), "446/447");
        assert_eq!(
            health_note(&both),
            " (camera gap: frame 7 absent in source; lidar gap: frames 177-180 absent in source)"
        );
        // A camera whose PNGs and lines could not be paired: unknown, not 0.
        let unpaired = DriveInfo {
            images: None,
            ..info(false, None, Vec::new())
        };
        assert_eq!(camera_cell(&unpaired), "?");
    }
}
