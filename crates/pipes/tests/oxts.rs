//! The third stream, end to end, with no dataset: the vehicle's own motion
//! (KITTI's OXTS rows) reaching the fusion, being subtracted, and being
//! accounted for.
//!
//! The scene is the fixture's line of points, so nothing here asserts a
//! velocity -- `pipes-kitti`'s tracker tests do that on a synthetic turn,
//! and the ignored real-drive harness on drive 0005. What a whole run adds,
//! and what these pin, is the plumbing: the driver's rows balance against
//! the frame slots, every row that reaches `track` owes an evidence row,
//! the summary says which frame the velocities are in, a frame the source
//! lacks is a declared substitute rather than a silent default, and the
//! stream can be turned off and refused.

use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{assert_ok, evidence, line_with, pipes, run_json, stderr, stdout, summary, u64_of};

/// Frames and sweeps per fixture run, and the period they share: 5 ms for
/// the unpaced runs, and 25 ms for the paced one, which a loaded host would
/// have to stall for a whole period to disturb.
const N: usize = 8;
const PERIOD_NS: i64 = 5_000_000;
const PACED_PERIOD_NS: i64 = 25_000_000;

/// 5 m/s forward, turning at 0.1 rad/s, on every row.
fn motion(_: usize) -> (f64, f64, f64) {
    (5.0, 0.0, 0.1)
}

/// A lidar fixture with a calibration and a GPS/IMU row per frame but those
/// of `absent`.
fn fixture(absent: &[usize]) -> FixtureDrive {
    fixture_at(absent, PERIOD_NS)
}

/// [`fixture`] at a period of its own.
fn fixture_at(absent: &[usize], period_ns: i64) -> FixtureDrive {
    let fx = FixtureDrive::with_velodyne(N, N, period_ns).unwrap();
    fx.write_calib().unwrap();
    fx.write_oxts(absent, motion).unwrap();
    fx
}

/// `summary.json`'s track block.
fn track_block(s: &serde_json::Value) -> serde_json::Value {
    s["lidar"]["reduce"]["detect"]["track"].clone()
}

#[test]
fn the_vehicle_s_motion_reaches_the_fusion_and_is_accounted_for() {
    // Paced, at 25 ms a frame. Unpaced, the GPS/IMU thread races the lidar
    // chain to the fusion, and a loaded host can hand a sweep over before its
    // row -- the unpaced mode's own truth, not the stream's. Paced, each row
    // is released a period before its sweep is admitted.
    let fx = fixture_at(&[], PACED_PERIOD_NS);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name", "ego", "--track", "on", "--cap", "16", "--rate", "1",
        ],
    );
    assert_ok(&o);
    let out = stdout(&o);
    // The driver's own line, and the edge's conservation, both OK.
    for want in [
        &format!("INVARIANT driver=oxts admitted={N} missing=0 n_rows={N} -> OK"),
        &format!("INVARIANT edge=oxts->track delivered={N} dropped=0 admitted={N} -> OK"),
        &format!("INVARIANT rows edge=oxts->track rows={N} admitted={N} -> OK"),
        &format!("oxts admitted={N} missing=0 n_rows={N}"),
    ] {
        assert!(
            out.lines().any(|l| l == want),
            "no line {want:?} in:\n{out}"
        );
    }
    // Every row the driver admitted left a row on the evidence, as every
    // sensor's do, and every one that reached the fusion left one too.
    let csv = evidence(&cwd, "ego");
    assert_eq!(csv.with("edge", "oxts").len(), N);
    assert!(csv
        .with("edge", "oxts")
        .iter()
        .all(|r| csv.col(r, "stage") == "oxts-driver" && csv.col(r, "outcome") == "Delivered"));
    assert_eq!(csv.with("edge", "oxts->track").len(), N);
    // The summary says what the stream was and what the tracker did with it.
    let s = summary(&cwd, "ego");
    assert_eq!(u64_of(&s["oxts"], "admitted"), N as u64);
    assert_eq!(u64_of(&s["oxts"], "missing"), 0);
    assert_eq!(u64_of(&s["oxts"], "delivered"), N as u64);
    assert_eq!(s["oxts"]["absent_in_source"], serde_json::json!([]));
    let t = track_block(&s);
    assert_eq!(t["velocity_frame"], serde_json::json!("over ground"));
    assert_eq!(
        u64_of(&t, "ego_compensated") + u64_of(&t, "ego_stale") + u64_of(&t, "ego_missing"),
        u64_of(&t, "delivered")
    );
    assert_eq!(u64_of(&t, "ego_missing"), 0);
    assert!(u64_of(&t, "ego_compensated") >= 1);
    assert!((t["ego_speed_mps_max"].as_f64().unwrap() - 5.0).abs() < 1e-9);
    assert!((t["ego_yaw_rate_rps_max"].as_f64().unwrap() - 0.1).abs() < 1e-9);
    assert!(
        t["gate_rationale"]
            .as_str()
            .unwrap()
            .contains("no ego-yaw term"),
        "{}",
        t["gate_rationale"]
    );
    // The run's own words.
    let ego = line_with(&out, "track ego motion: ");
    assert!(ego.contains("velocities over ground"), "{ego}");
    assert!(
        ego.contains("speed up to 5.00 m/s, yaw rate up to 0.100 rad/s"),
        "{ego}"
    );
    // And the provenance.
    let r = run_json(&cwd, "ego");
    assert_eq!(r["oxts"], serde_json::json!("auto"));
    assert_eq!(r["n_oxts"], serde_json::json!(N));
}

#[test]
fn a_frame_without_a_row_is_tracked_from_the_newest_earlier_one_and_says_so() {
    // Frames 3 to 5 have no row: 3 and 4 are within two frames of row 2,
    // which stands in; 5 is not, and goes uncompensated.
    let fx = fixture(&[3, 4, 5]);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "gap", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert!(
        out.lines().any(|l| l
            == "INVARIANT driver=oxts admitted=5 missing=3 frames=8 = n_rows 5 + absent_in_source 3 (rows 3) -> OK"),
        "{out}"
    );
    // Said before the clock started, and once as an event when the replay
    // reached it.
    assert!(
        out.contains("no GPS/IMU row in the source for frames 3-5"),
        "{out}"
    );
    let events = std::fs::read_to_string(cwd.path().join("runs/gap/events.csv")).unwrap();
    assert!(
        events.contains("SourceGap,oxts-driver,") && events.contains("gps/imu frames 3-5"),
        "{events}"
    );
    let csv = evidence(&cwd, "gap");
    let absent: Vec<String> = csv
        .with("edge", "oxts")
        .iter()
        .filter(|r| csv.col(r, "reason") == "absent_in_source")
        .map(|r| csv.col(r, "seq"))
        .collect();
    assert_eq!(absent, ["3", "4", "5"]);
    let s = summary(&cwd, "gap");
    assert_eq!(s["oxts"]["absent_in_source"], serde_json::json!([3, 4, 5]));
    let t = track_block(&s);
    assert_eq!(t["velocity_frame"], serde_json::json!("over ground"));
    // Unpaced, the lidar's two-deep queue can evict sweeps on a loaded host,
    // so the expectation is over the sweeps the tracker was actually handed:
    // 3 and 4 from row 2, 5 from nothing, every other from its own row.
    let reached: Vec<u64> = csv
        .with("edge", "obj->track")
        .iter()
        .filter(|r| csv.col(r, "outcome") == "Delivered")
        .map(|r| csv.col(r, "seq").parse().unwrap())
        .collect();
    let stale = reached.iter().filter(|s| [3, 4].contains(s)).count() as u64;
    let missing = reached.iter().filter(|&&s| s == 5).count() as u64;
    assert_eq!(u64_of(&t, "ego_stale"), stale, "{reached:?}");
    assert_eq!(u64_of(&t, "ego_missing"), missing, "{reached:?}");
    assert_eq!(
        u64_of(&t, "ego_compensated"),
        reached.len() as u64 - stale - missing,
        "{reached:?}"
    );
}

#[test]
fn the_stream_can_be_turned_off_and_is_refused_where_nothing_reads_it() {
    let fx = fixture(&[]);
    let cwd = TempDir::new().unwrap();
    // Off: the run is what it was before the stream existed, and says so.
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name", "off", "--track", "on", "--cap", "16", "--oxts", "off",
        ],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert!(
        !out.lines().any(|l| l.starts_with("oxts ")
            || l.contains("driver=oxts")
            || l.contains("oxts->track")),
        "{out}"
    );
    let ego = line_with(&out, "track ego motion: ");
    assert!(ego.contains("none (no GPS/IMU stream)"), "{ego}");
    let s = summary(&cwd, "off");
    assert!(s["oxts"].is_null());
    let t = track_block(&s);
    assert_eq!(
        t["velocity_frame"],
        serde_json::json!("relative to the vehicle")
    );
    assert_eq!(u64_of(&t, "ego_compensated"), 0);
    assert!(t["ego_speed_mps_max"].is_null());
    assert!(t["gate_rationale"]
        .as_str()
        .unwrap()
        .contains("ego-yaw swing"));
    assert_eq!(run_json(&cwd, "off")["oxts"], serde_json::json!("off"));
    assert!(run_json(&cwd, "off")["n_oxts"].is_null());
    // On, with no fusion to read it: refused before the clock starts.
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "refused", "--track", "off", "--oxts", "on"],
    );
    assert!(!o.status.success(), "{}", stdout(&o));
    assert!(
        stderr(&o).contains("--oxts on needs the fusion"),
        "{}",
        stderr(&o)
    );
    assert!(!cwd.path().join("runs/refused/_COMPLETE").exists());
}
