//! `pipes drives` through the real binary.
//!
//! `crates/pipes` is binary-only, so the subcommand is exercised the way a
//! user meets it: `env!("CARGO_BIN_EXE_pipes")`, a synthetic data root, and
//! assertions on stdout and the exit code. `tests/common/` is not used here
//! because every helper in it prefixes `run`.
//!
//! The exit code is half the contract — "0 if the root exists, even with zero
//! drives; 1 only if the root itself is missing" — so every test checks it.

use std::process::{Command, Output};

use pipes_kitti::testing::{
    write_fixture_at, write_velo_fixture_at, FixtureDrive, FIXTURE_DATE, FIXTURE_DRIVE, FIXTURE_H,
    FIXTURE_PERIOD_NS, FIXTURE_W,
};
use tempfile::TempDir;

const OTHER_DATE: &str = "2011_09_30";
const OTHER_DRIVE: &str = "2011_09_30_drive_0028_sync";

fn drives(root: &std::path::Path, args: &[&str]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_pipes"));
    c.arg("drives")
        .arg("--kitti-root")
        .arg(root)
        // The developer's own dataset must not leak in through the environment.
        .env_remove("PIPES_KITTI_ROOT")
        .args(args);
    c.output().expect("spawn pipes")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn assert_code(o: &Output, code: i32) {
    assert_eq!(
        o.status.code(),
        Some(code),
        "--- stdout ---\n{}\n--- stderr ---\n{}",
        stdout(o),
        stderr(o)
    );
}

fn json(o: &Output) -> Vec<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(&stdout(o))
        .unwrap_or_else(|e| panic!("--json did not print JSON ({e}):\n{}", stdout(o)))
        .as_array()
        .expect("--json prints an array")
        .clone()
}

#[test]
fn human_output_names_the_drive_its_frames_and_its_resolution() {
    let fx = FixtureDrive::new(16).unwrap();
    let o = drives(fx.root(), &[]);
    assert_code(&o, 0);
    let out = stdout(&o);
    let row = out
        .lines()
        .find(|l| l.contains(FIXTURE_DRIVE))
        .unwrap_or_else(|| panic!("no row for {FIXTURE_DRIVE}:\n{out}"));
    for needle in [
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        "16",
        &format!("{FIXTURE_W}x{FIXTURE_H}"),
        "ok",
    ] {
        assert!(row.contains(needle), "row {row:?} has no {needle:?}");
    }
    assert!(
        out.lines().next().is_some_and(|l| l.contains("resolution")),
        "the first line must be a header:\n{out}"
    );
}

#[test]
fn json_output_has_the_twelve_contract_keys() {
    let fx = FixtureDrive::new(16).unwrap();
    let o = drives(fx.root(), &["--json"]);
    assert_code(&o, 0);
    let rows = json(&o);
    assert_eq!(rows.len(), 1, "{rows:?}");
    let r = &rows[0];
    assert_eq!(r["date"], serde_json::json!(FIXTURE_DATE));
    assert_eq!(r["drive"], serde_json::json!(FIXTURE_DRIVE));
    assert_eq!(r["frames"], serde_json::json!(16));
    assert_eq!(r["width"], serde_json::json!(FIXTURE_W));
    assert_eq!(r["height"], serde_json::json!(FIXTURE_H));
    assert_eq!(r["images"], serde_json::json!(16));
    assert_eq!(r["lidar"], serde_json::json!(false));
    assert_eq!(r["sweeps"], serde_json::Value::Null);
    assert_eq!(r["lidar_absent_in_source"], serde_json::json!([]));
    assert_eq!(r["camera_absent_in_source"], serde_json::json!([]));
    assert_eq!(r["ok"], serde_json::json!(true));
    assert_eq!(r["problem"], serde_json::json!(""));
    // Nothing extra: a consumer that switches on the key set must not have to
    // guess which fields it may ignore.
    let keys: Vec<&str> = r
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys.len(), 12, "{keys:?}");
}

/// The listing is how a user finds which drives carry a second stream, which
/// is the question `--lidar auto` answers silently.
///
/// Both drives are in one listing on purpose: a `lidar` column that was
/// hard-coded either way would pass half of this.
#[test]
fn the_listing_says_which_drives_carry_lidar() {
    let tmp = TempDir::new().unwrap();
    for drive in [FIXTURE_DRIVE, OTHER_DRIVE] {
        write_fixture_at(tmp.path(), FIXTURE_DATE, drive, 8, FIXTURE_PERIOD_NS).unwrap();
    }
    write_velo_fixture_at(tmp.path(), FIXTURE_DATE, OTHER_DRIVE, 8, FIXTURE_PERIOD_NS).unwrap();

    let rows = json(&drives(tmp.path(), &["--json"]));
    let of = |name: &str| {
        rows.iter()
            .find(|r| r["drive"] == serde_json::json!(name))
            .unwrap_or_else(|| panic!("{name} missing from {rows:?}"))
            .clone()
    };
    assert_eq!(of(FIXTURE_DRIVE)["lidar"], serde_json::json!(false));
    assert_eq!(of(OTHER_DRIVE)["lidar"], serde_json::json!(true));
    // Sweeps, read the way `run` reads them, against the drive's frames.
    assert_eq!(of(FIXTURE_DRIVE)["sweeps"], serde_json::Value::Null);
    assert_eq!(of(OTHER_DRIVE)["sweeps"], serde_json::json!(8));
    // A camera-only drive is healthy, not broken: many KITTI downloads are
    // `image_02` alone, and marking them broken would make the column useless.
    for name in [FIXTURE_DRIVE, OTHER_DRIVE] {
        assert_eq!(of(name)["ok"], serde_json::json!(true), "{name}");
    }

    // And the human table carries it too, under a `lidar` header, beside
    // the camera's own count.
    let o = drives(tmp.path(), &[]);
    assert_code(&o, 0);
    let out = stdout(&o);
    let header = out.lines().next().expect("a header");
    assert_eq!(
        cells(header),
        [
            "date",
            "drive",
            "frames",
            "resolution",
            "camera",
            "lidar",
            "health"
        ],
        "{header:?}"
    );
    let row = |name: &str| {
        out.lines().find(|l| l.contains(name)).unwrap_or_else(|| {
            panic!(
                "no row for {name}:
{out}"
            )
        })
    };
    // Sweeps against frames where there is a lidar; `-` and a word saying
    // so where there is none, so the camera-only drive reads as a fact.
    assert_eq!(cells(row(OTHER_DRIVE))[4..6], ["8/8", "8/8"]);
    assert!(
        !row(OTHER_DRIVE).contains("camera-only"),
        "{}",
        row(OTHER_DRIVE)
    );
    assert_eq!(cells(row(FIXTURE_DRIVE))[4..6], ["8/8", "-"]);
    assert!(
        row(FIXTURE_DRIVE).ends_with("ok (camera-only)"),
        "{}",
        row(FIXTURE_DRIVE)
    );
}

/// A table line's cells, split on whitespace: the first six are `date`,
/// `drive`, `frames`, `resolution`, `camera` and `lidar`, which never hold a
/// space; the health words follow.
fn cells(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

/// The reason the subcommand exists. A user with a half-unzipped second scene
/// needs to *see* it; a listing that failed, or that quietly dropped the bad
/// row, would send them back to guessing.
#[test]
fn a_broken_drive_is_reported_and_the_listing_still_exits_zero() {
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
    let data = tmp
        .path()
        .join(OTHER_DATE)
        .join(OTHER_DRIVE)
        .join("image_02")
        .join("data");
    std::fs::rename(data.join("0000000005.png"), data.join("oops.png")).unwrap();

    let o = drives(tmp.path(), &["--json"]);
    assert_code(&o, 0);
    let rows = json(&o);
    assert_eq!(
        rows.len(),
        2,
        "the broken drive must still be listed: {rows:?}"
    );

    let broken = rows
        .iter()
        .find(|r| r["drive"] == serde_json::json!(OTHER_DRIVE))
        .expect("the broken drive");
    assert_eq!(broken["ok"], serde_json::json!(false));
    let problem = broken["problem"].as_str().expect("problem is a string");
    assert!(
        problem.contains("oops.png"),
        "problem must name the file that is not a frame: {problem}"
    );

    // Positive control: the untouched drive in the same root is still sound
    // and still carries an empty `problem`, so `ok:false` above is a verdict
    // and not the listing giving up.
    let sound = rows
        .iter()
        .find(|r| r["drive"] == serde_json::json!(FIXTURE_DRIVE))
        .expect("the sound drive");
    assert_eq!(sound["ok"], serde_json::json!(true));
    assert_eq!(sound["problem"], serde_json::json!(""));

    // The human table must mark it too, or the default output is the one that
    // hides the problem.
    let o = drives(tmp.path(), &[]);
    assert_code(&o, 0);
    let out = stdout(&o);
    let row = out
        .lines()
        .find(|l| l.contains(OTHER_DRIVE))
        .unwrap_or_else(|| panic!("no row for {OTHER_DRIVE}:\n{out}"));
    assert!(row.contains("BROKEN"), "row {row:?} carries no marker");
    assert!(
        out.lines()
            .find(|l| l.contains(FIXTURE_DRIVE))
            .is_some_and(|l| !l.contains("BROKEN")),
        "the sound drive was marked broken too:\n{out}"
    );
}

#[test]
fn an_empty_root_says_so_and_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let o = drives(tmp.path(), &[]);
    assert_code(&o, 0);
    let out = stdout(&o);
    assert!(
        out.contains("no drives found under") && out.contains(&tmp.path().display().to_string()),
        "an empty root must name itself:\n{out}"
    );

    // …and `--json` says the same thing in its own shape: an empty array, not
    // an error and not nothing at all.
    let o = drives(tmp.path(), &["--json"]);
    assert_code(&o, 0);
    assert!(json(&o).is_empty());
}

#[test]
fn a_missing_root_exits_one() {
    let tmp = TempDir::new().unwrap();
    let absent = tmp.path().join("no_such_root");
    let o = drives(&absent, &[]);
    assert_code(&o, 1);
    let err = stderr(&o);
    assert!(
        err.contains(&absent.display().to_string()),
        "the error must name the root it could not read:\n{err}"
    );
    assert!(
        err.contains("--kitti-root"),
        "the error must name the override:\n{err}"
    );
    assert!(stdout(&o).is_empty(), "stdout: {}", stdout(&o));

    // Positive control: the same binary, the same flags, a root that exists —
    // exit 0. Without it, "exit 1" would also be satisfied by a subcommand
    // that never worked at all.
    assert_code(&drives(tmp.path(), &[]), 0);
}

/// A gap in a sensor's source is shown per drive, as a fact beside `ok`:
/// each sensor's samples against the drive's frames in its own column, the
/// frames named in the health column, and counted on the last line. The same
/// in `--json`.
#[test]
fn the_listing_names_the_frames_a_source_has_no_sample_for() {
    let fx = FixtureDrive::with_gaps(10, &[], &[3, 4], FIXTURE_PERIOD_NS).unwrap();
    let o = drives(fx.root(), &[]);
    assert_code(&o, 0);
    let out = stdout(&o);
    let row = out
        .lines()
        .find(|l| l.contains(FIXTURE_DRIVE))
        .unwrap_or_else(|| panic!("no row:\n{out}"));
    assert_eq!(cells(row)[4..6], ["10/10", "8/10"], "{row}");
    assert!(
        row.ends_with("ok (lidar gap: frames 3-4 absent in source)"),
        "{row}"
    );
    assert!(
        out.lines()
            .last()
            .is_some_and(|l| l.ends_with(", 0 broken, 1 with a gap in the source")),
        "{out}"
    );
    let rows = json(&drives(fx.root(), &["--json"]));
    assert_eq!(rows[0]["sweeps"], serde_json::json!(8));
    assert_eq!(rows[0]["images"], serde_json::json!(10));
    assert_eq!(rows[0]["frames"], serde_json::json!(10));
    assert_eq!(rows[0]["lidar_absent_in_source"], serde_json::json!([3, 4]));
    assert_eq!(rows[0]["ok"], serde_json::json!(true));

    // The camera's gap, on the same terms.
    let fx = FixtureDrive::with_gaps(10, &[6], &[], FIXTURE_PERIOD_NS).unwrap();
    let out = stdout(&drives(fx.root(), &[]));
    let row = out.lines().find(|l| l.contains(FIXTURE_DRIVE)).unwrap();
    assert_eq!(cells(row)[4..6], ["9/10", "10/10"], "{row}");
    assert!(
        row.ends_with("ok (camera gap: frame 6 absent in source)"),
        "{row}"
    );
    let rows = json(&drives(fx.root(), &["--json"]));
    assert_eq!(rows[0]["images"], serde_json::json!(9));
    assert_eq!(rows[0]["frames"], serde_json::json!(10));
    assert_eq!(rows[0]["camera_absent_in_source"], serde_json::json!([6]));

    // A camera that stops before the lidar: the drive has the lidar's ten
    // frames, and the camera's last two are a gap in ITS source.
    let fx = FixtureDrive::with_velodyne(8, 10, FIXTURE_PERIOD_NS).unwrap();
    let out = stdout(&drives(fx.root(), &[]));
    let row = out.lines().find(|l| l.contains(FIXTURE_DRIVE)).unwrap();
    assert_eq!(cells(row)[4..6], ["8/10", "10/10"], "{row}");
    assert!(
        row.ends_with("ok (camera gap: frames 8-9 absent in source)"),
        "{row}"
    );
    let rows = json(&drives(fx.root(), &["--json"]));
    assert_eq!(rows[0]["frames"], serde_json::json!(10));
    assert_eq!(
        rows[0]["camera_absent_in_source"],
        serde_json::json!([8, 9])
    );
}
