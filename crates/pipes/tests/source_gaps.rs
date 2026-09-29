//! Frames missing from the source, on either sensor, end to end: replayed as
//! missing inputs rather than refused, every frame slot accounted for, the
//! streams kept in step because pairing is by time, and the fusion saying
//! which instants had no partner.
//!
//! The fixtures are laid out the way KITTI lays out a gap (drive 0009's lidar
//! has no sweeps for frames 177-180): a file named by its frame for every
//! frame that exists, and one timestamp line per frame, line `i + 1` for frame
//! `i`, left blank where the frame has no file. The last test replays drive
//! 0009 itself, when it is on disk.
//!
//! Unpaced runs may evict on the chain's fixed-cap Drop* edges on a loaded
//! host, so what is pinned there is conservation and attribution, computed
//! from the evidence; the paced tests pin the schedule itself.

use std::path::{Path, PathBuf};

use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{assert_ok, evidence, pipes, stdout, summary, u64_of, Csv};

/// Frame numbers per fixture, and the period they share.
const FRAMES: usize = 10;
const PERIOD_NS: i64 = 5_000_000;

/// The period of the paced tests: 25 ms, as `lidar.rs` and `track.rs` give
/// theirs, so a scheduler stall cannot pass for the schedule.
const PACED_PERIOD_NS: i64 = 25_000_000;

/// A fixture with its calibration, so `--track on` runs the fusion.
fn gapped(camera_absent: &[usize], lidar_absent: &[usize], period_ns: i64) -> FixtureDrive {
    let fx = FixtureDrive::with_gaps(FRAMES, camera_absent, lidar_absent, period_ns).unwrap();
    fx.write_calib().unwrap();
    fx
}

fn invariants(out: &str) -> Vec<&str> {
    out.lines()
        .filter(|l| l.starts_with("INVARIANT "))
        .collect()
}

/// Every INVARIANT line says OK, and there are some.
fn assert_all_invariants_ok(out: &str) {
    let inv = invariants(out);
    assert!(!inv.is_empty(), "no invariant lines:\n{out}");
    for l in inv {
        assert!(l.ends_with("-> OK"), "{l}\n--- stdout ---\n{out}");
    }
}

/// The driver rows of one sensor's pseudo-edge: `(seq, outcome, reason,
/// tov_start_ns, due_ns, arrival_ns)`, in seq order.
fn driver_rows(csv: &Csv, edge: &str) -> Vec<(u64, String, String, String, String, i64)> {
    let mut v: Vec<_> = csv
        .with("edge", edge)
        .iter()
        .map(|r| {
            (
                csv.col(r, "seq").parse().expect("seq"),
                csv.col(r, "outcome"),
                csv.col(r, "reason"),
                csv.col(r, "tov_start_ns"),
                csv.col(r, "due_ns"),
                csv.col(r, "arrival_ns").parse().expect("arrival_ns"),
            )
        })
        .collect();
    v.sort_by_key(|r| r.0);
    v
}

/// The seqs of one pseudo-edge's rows with reason `absent_in_source`.
fn absent_seqs(csv: &Csv, edge: &str) -> Vec<u64> {
    driver_rows(csv, edge)
        .into_iter()
        .filter(|r| r.2 == "absent_in_source")
        .map(|r| r.0)
        .collect()
}

/// Every delivered camera frame and sweep carries the seq of the frame it
/// belongs to: its instant (a sweep's start) is that frame's place on the
/// fixture's clock. A frame of one number and a sweep of the same number are
/// then the same instant, which is what keeps a gap on one stream from
/// shifting the other's numbers.
fn assert_frames_keep_their_numbers(csv: &Csv, period_ns: i64) {
    for edge in ["cam0", "velo"] {
        for r in driver_rows(csv, edge).iter().filter(|r| r.1 == "Delivered") {
            let tov: i64 = r.3.parse().expect("tov_start_ns");
            assert_eq!(
                tov,
                pipes_kitti::testing::FIXTURE_BASE_NS + r.0 as i64 * period_ns,
                "{edge} seq {} carries another frame's instant",
                r.0
            );
        }
    }
}

fn line_starting<'a>(out: &'a str, prefix: &str) -> &'a str {
    out.lines()
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("no line starting {prefix:?}:\n{out}"))
}

/// The lidar's source lacks frames 3 and 4. The drive opens; every sweep
/// keeps its frame number; the two absent frames are `Missing` rows with the
/// reason that names the cause and no time of validity; all ten slots are
/// accounted for; and the fusion says the camera frames of those instants had
/// no sweep to pair with.
#[test]
fn a_lidar_gap_is_recorded_as_missing_input_and_every_slot_is_accounted_for() {
    let fx = gapped(&[], &[3, 4], PERIOD_NS);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "lgap", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert_all_invariants_ok(&out);
    // Said before the clock, and at the end.
    assert!(
        line_starting(&out, "velo ").contains(
            "frames 3-4 (2 frames, 10.0 ms with no measurement between frame 2 and frame 5)"
        ),
        "{out}"
    );
    assert!(
        out.lines().any(|l| l
            == "velo admitted=8 missing=2 n_sweeps=8 frames=10 absent_in_source=2 (frames 3-4: no sweep in the source)"),
        "{out}"
    );
    assert!(
        out.lines().any(|l| l
            == "INVARIANT driver=velo admitted=8 missing=2 frames=10 = n_sweeps 8 + absent_in_source 2 (rows 2) -> OK"),
        "{out}"
    );

    let csv = evidence(&cwd, "lgap");
    let velo = driver_rows(&csv, "velo");
    assert_eq!(velo.len(), FRAMES, "one driver row per frame slot");
    let delivered: Vec<u64> = velo
        .iter()
        .filter(|r| r.1 == "Delivered")
        .map(|r| r.0)
        .collect();
    assert_eq!(
        delivered,
        vec![0, 1, 2, 5, 6, 7, 8, 9],
        "a sweep's seq is its frame number"
    );
    let absent: Vec<_> = velo.iter().filter(|r| r.1 == "Missing").collect();
    assert_eq!(absent.len(), 2, "{absent:?}");
    for r in &absent {
        assert!([3, 4].contains(&r.0), "{r:?}");
        assert_eq!(r.2, "absent_in_source");
        assert_eq!(r.3, "0", "an instant was invented for frame {}", r.0);
    }
    // The camera has every frame, and is untouched by the lidar's gap.
    assert!(absent_seqs(&csv, "cam0").is_empty());
    assert_eq!(driver_rows(&csv, "cam0").len(), FRAMES);

    // Every frame slot of the lidar stream against the fusion: the two
    // absent ones in a named bucket of their own, and the losses adding up.
    let s = summary(&cwd, "lgap");
    assert_eq!(s["lidar"]["frames"], serde_json::json!(FRAMES));
    assert_eq!(s["lidar"]["absent_in_source"], serde_json::json!([3, 4]));
    let t = &s["lidar"]["reduce"]["detect"]["track"];
    let b = &t["sweeps_before_track"];
    assert_eq!(u64_of(b, "absent_in_source"), 2, "{b}");
    assert_eq!(u64_of(b, "sweeps"), 8, "{b}");
    let sets = line_starting(&out, "fusion sets: ");
    assert!(
        sets.contains(&format!(
            "NEVER REACHED track {} of 10 frames (8 sweeps + 2 absent in source) = ",
            10 - u64_of(b, "reached")
        )) && sets.ends_with(" + absent in source 2"),
        "{sets}"
    );
    assert!(!sets.contains("MISMATCH"), "{sets}");
    // The instants with no partner, said: frames 3-4's camera frames had no
    // sweep to pair with, and no set used them.
    let gap = &t["source_gaps"];
    assert_eq!(gap["no_sweep"], serde_json::json!([3, 4]));
    assert_eq!(gap["no_sweep_camera_used_stale"], serde_json::json!([]));
    let reached = gap["no_sweep_camera_reached_track"]
        .as_array()
        .expect("an array")
        .len();
    assert!(
        out.lines().any(|l| l
            == format!("fusion sets for frames 3-4: none, no sweep in the lidar's source -- camera frames that reached track and had no sweep to pair with: {reached} of 2")),
        "{out}"
    );
    // Pairing by time held across the gap: every set named a frame inside its
    // own sweep, and none named frame 3 or 4.
    assert_eq!(t["cam_pair_join_ok"], serde_json::json!(true), "{t}");

    // The gap is one WARN event, naming the frames and the cause.
    let events = std::fs::read_to_string(cwd.path().join("runs/lgap/events.csv")).unwrap();
    let gap_events: Vec<&str> = events.lines().filter(|l| l.contains("SourceGap")).collect();
    assert_eq!(gap_events.len(), 1, "{events}");
    assert!(
        gap_events[0].contains("lidar frames 3-4") && gap_events[0].contains("absent_in_source"),
        "{}",
        gap_events[0]
    );
}

/// A lidar that stops before the camera does: its timestamp files end at
/// line 8. The camera declares ten frames, so the lidar's last two are absent
/// in its source too -- the lidar's own files could not show it -- and they
/// are replayed and counted like any other gap.
#[test]
fn a_lidar_that_stops_early_is_missing_its_last_frames() {
    let fx = FixtureDrive::with_velodyne(FRAMES, 8, PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    assert_lidar_tail(&fx, "tail");
}

/// The same tail declared by the lidar itself: its timestamp files have ten
/// lines, the last two blank. Read exactly as the tail above.
#[test]
fn trailing_blank_lines_on_the_lidar_are_frames_absent_in_its_source() {
    let fx = gapped(&[], &[8, 9], PERIOD_NS);
    let text = std::fs::read_to_string(fx.velo_dir().join("timestamps.txt")).unwrap();
    assert_eq!(text.lines().count(), FRAMES, "{text:?}");
    assert!(
        text.ends_with("\n\n\n"),
        "two trailing blank lines: {text:?}"
    );
    assert_lidar_tail(&fx, "tailblank");
}

/// A lidar with no sweep for frames 8-9 of ten, run: two absent rows at the
/// end, every slot accounted for.
fn assert_lidar_tail(fx: &FixtureDrive, name: &str) {
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, fx, &["--name", name, "--track", "on", "--cap", "16"]);
    assert_ok(&o);
    let out = stdout(&o);
    assert_all_invariants_ok(&out);
    assert!(
        line_starting(&out, "velo ")
            .contains("frames 8-9 (2 frames, after the last sample, frame 7)"),
        "{out}"
    );
    assert!(
        out.lines().any(|l| l
            == "velo admitted=8 missing=2 n_sweeps=8 frames=10 absent_in_source=2 (frames 8-9: no sweep in the source)"),
        "{out}"
    );
    let csv = evidence(&cwd, name);
    assert_eq!(driver_rows(&csv, "velo").len(), FRAMES);
    assert_eq!(absent_seqs(&csv, "velo"), vec![8, 9]);
    let s = summary(&cwd, name);
    assert_eq!(s["lidar"]["frames"], serde_json::json!(FRAMES));
    assert_eq!(s["lidar"]["absent_in_source"], serde_json::json!([8, 9]));
    assert_eq!(s["n_frames"], serde_json::json!(FRAMES));
}

/// A camera that stops before the lidar does: `timestamps.txt` ends at line
/// 8, and the lidar declares ten frames. The drive has ten, so the camera's
/// last two are absent in ITS source -- recorded, not blamed on the pipeline:
/// the sweeps of those instants expire as `pair_absent_in_source`, never as
/// `pair_late`.
#[test]
fn a_camera_that_stops_early_is_missing_its_last_frames_and_the_sweeps_say_why() {
    let fx = FixtureDrive::with_velodyne(8, FRAMES, PACED_PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    assert_camera_tail(&fx, "ctail");
}

/// The same tail declared by the camera itself: `timestamps.txt` has ten
/// lines, the last two blank.
#[test]
fn trailing_blank_lines_on_the_camera_are_frames_absent_in_its_source() {
    let fx = gapped(&[8, 9], &[], PACED_PERIOD_NS);
    let text = std::fs::read_to_string(fx.image_dir().join("timestamps.txt")).unwrap();
    assert_eq!(text.lines().count(), FRAMES, "{text:?}");
    assert_camera_tail(&fx, "ctailblank");
}

/// A camera with no frame 8 or 9 of ten, run paced with the fusion: two
/// absent rows at the end, the slots accounted for, and the sweeps of those
/// instants expired for the camera's source and for nothing else.
fn assert_camera_tail(fx: &FixtureDrive, name: &str) {
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        fx,
        &[
            "--name", name, "--track", "on", "--rate", "1.0", "--cap", "16",
        ],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert_all_invariants_ok(&out);
    assert!(
        line_starting(&out, "cam0 ")
            .contains("frames 8-9 (2 frames, after the last sample, frame 7)"),
        "{out}"
    );
    assert!(
        out.lines().any(|l| {
            l
            == "cam0 on_disk=8 n_frames=10 absent_in_source=2 (frames 8-9: no frame in the source)"
        }),
        "{out}"
    );
    assert!(
        out.lines()
            .any(|l| l.starts_with("INVARIANT driver admitted=")
                && l.ends_with("n_frames=10 = on disk 8 + absent_in_source 2 (rows 2) -> OK")),
        "{out}"
    );
    let csv = evidence(&cwd, name);
    assert_eq!(driver_rows(&csv, "cam0").len(), FRAMES, "one row per slot");
    assert_eq!(absent_seqs(&csv, "cam0"), vec![8, 9]);
    assert!(absent_seqs(&csv, "velo").is_empty());
    assert_frames_keep_their_numbers(&csv, PACED_PERIOD_NS);
    // The sweeps of frames 8 and 9 reached the fusion -- paced, nothing in
    // front of it evicts -- and expired for the camera's source.
    let track: Vec<(u64, String)> = csv
        .rows
        .iter()
        .filter(|r| csv.col(r, "edge") == "track" && csv.col(r, "outcome") == "Missing")
        .map(|r| {
            (
                csv.col(r, "seq").parse().expect("seq"),
                csv.col(r, "reason"),
            )
        })
        .collect();
    assert!(
        !track
            .iter()
            .any(|(_, r)| r == "pair_late" || r == "pair_dropped"),
        "the camera's missing tail was blamed on the pipeline: {track:?}"
    );
    let s = summary(&cwd, name);
    assert_eq!(s["n_frames"], serde_json::json!(FRAMES));
    assert_eq!(s["absent_in_source"], serde_json::json!([8, 9]));
    let t = &s["lidar"]["reduce"]["detect"]["track"];
    assert_eq!(t["source_gaps"]["no_frame"], serde_json::json!([8, 9]));
    assert_eq!(
        t["expired_absent_in_source"],
        serde_json::json!([8, 9]),
        "{t}"
    );
    assert_eq!(
        track,
        vec![
            (8, "pair_absent_in_source".to_string()),
            (9, "pair_absent_in_source".to_string())
        ]
    );
    assert!(
        out.lines().any(|l| l
            == "fusion sets for frames 8-9: no camera frame in the source -- sweeps that reached track and expired with no frame to pair (pair_absent_in_source): 2 of 2"),
        "{out}"
    );
    assert_eq!(t["cam_pair_join_ok"], serde_json::json!(true), "{t}");
    // Named once, as a WARN event, by the camera driver.
    let events =
        std::fs::read_to_string(cwd.path().join("runs").join(name).join("events.csv")).unwrap();
    assert!(
        events
            .lines()
            .any(|l| l.contains("SourceGap") && l.contains("camera frames 8-9")),
        "{events}"
    );
}

/// The replay sits through the hole the source left rather than closing it:
/// paced, sweep 5 falls due three periods after sweep 2 and is not handed
/// over before then, and the absent frames are reported at their own places
/// in between, with derived deadlines and no instant.
#[test]
fn a_lidar_gap_leaves_a_hole_in_the_schedule() {
    let fx = gapped(&[], &[3, 4], PACED_PERIOD_NS);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "hole", "--rate", "1.0", "--cap", "16"],
    );
    assert_ok(&o);
    assert_all_invariants_ok(&stdout(&o));
    let csv = evidence(&cwd, "hole");
    let velo = driver_rows(&csv, "velo");
    let row = |f: u64| velo.iter().find(|r| r.0 == f).unwrap().clone();
    let due = |f: u64| -> i64 {
        row(f)
            .4
            .parse()
            .unwrap_or_else(|e| panic!("frame {f} has no deadline: {e}"))
    };
    for f in [2u64, 5] {
        assert_eq!(row(f).1, "Delivered", "sweep {f}: {:?}", row(f));
    }
    // The schedule: three periods from sweep 2's deadline to sweep 5's.
    assert_eq!(due(5) - due(2), 3 * PACED_PERIOD_NS);
    // Nothing handed over early: sweep 5 arrives no sooner than its deadline.
    assert!(row(5).5 >= due(5), "{:?}", row(5));
    // The absent frames fall due inside the hole, in order, a period apart,
    // and are reported no sooner than that.
    assert_eq!(due(3) - due(2), PACED_PERIOD_NS);
    assert_eq!(due(4) - due(3), PACED_PERIOD_NS);
    for f in [3u64, 4] {
        let r = row(f);
        assert_eq!(
            (r.1.as_str(), r.2.as_str(), r.3.as_str()),
            ("Missing", "absent_in_source", "0")
        );
        assert!(r.5 >= due(f), "frame {f} reported before its slot: {r:?}");
    }
}

/// The camera's source lacks frame 6. The camera replays it as a missing
/// input, the lidar is untouched, and the sweep of that instant -- which
/// reaches the fusion with no camera frame inside its range and none coming
/// -- expires with the cause named, rather than as a dropped or late frame.
#[test]
fn a_camera_gap_is_recorded_and_the_sweep_of_that_instant_says_why_it_has_no_pair() {
    let fx = gapped(&[6], &[], PACED_PERIOD_NS);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name", "cgap", "--track", "on", "--rate", "1.0", "--cap", "16",
        ],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert_all_invariants_ok(&out);
    assert!(
        out.lines().any(|l| l
            == "cam0 on_disk=9 n_frames=10 absent_in_source=1 (frame 6: no frame in the source)"),
        "{out}"
    );
    assert!(
        out.lines()
            .any(|l| l.starts_with("INVARIANT driver admitted=")
                && l.ends_with("n_frames=10 = on disk 9 + absent_in_source 1 (rows 1) -> OK")),
        "{out}"
    );
    let csv = evidence(&cwd, "cgap");
    assert_eq!(absent_seqs(&csv, "cam0"), vec![6]);
    assert!(absent_seqs(&csv, "velo").is_empty());
    let cam = driver_rows(&csv, "cam0");
    assert_eq!(cam.len(), FRAMES);
    let r6 = cam.iter().find(|r| r.0 == 6).unwrap();
    assert_eq!(r6.3, "0", "an instant was invented for camera frame 6");
    assert_frames_keep_their_numbers(&csv, PACED_PERIOD_NS);

    let s = summary(&cwd, "cgap");
    assert_eq!(s["absent_in_source"], serde_json::json!([6]));
    let t = &s["lidar"]["reduce"]["detect"]["track"];
    // Sweep 6 reached the fusion -- paced, nothing in front of it evicts --
    // and expired for its missing frame and for nothing else; no other sweep
    // did.
    let expired_src = t["expired_absent_in_source"].as_array().expect("array");
    assert_eq!(t["expired_absent_in_source"], serde_json::json!([6]), "{t}");
    let track_missing: Vec<String> = csv
        .rows
        .iter()
        .filter(|r| csv.col(r, "edge") == "track" && csv.col(r, "outcome") == "Missing")
        .map(|r| csv.col(r, "reason"))
        .collect();
    assert_eq!(
        track_missing
            .iter()
            .filter(|r| r.as_str() == "pair_absent_in_source")
            .count(),
        expired_src.len(),
        "{track_missing:?}"
    );
    assert!(
        !track_missing.iter().any(|r| r == "pair_dropped"),
        "a frame the source never had was blamed on a drop: {track_missing:?}"
    );
    assert_eq!(t["source_gaps"]["no_frame"], serde_json::json!([6]));
    assert!(
        out.lines().any(|l| l == format!(
            "fusion sets for frame 6: no camera frame in the source -- sweeps that reached track and expired with no frame to pair (pair_absent_in_source): {} of 1",
            expired_src.len()
        )),
        "{out}"
    );
    if !expired_src.is_empty() {
        assert!(
            line_starting(&out, "fusion sets: ").contains(" + camera frame absent in source 1"),
            "{out}"
        );
    }
    // Pairing by time re-locked after the gap: the sweeps either side paired
    // their own frames.
    assert_eq!(t["cam_pair_join_ok"], serde_json::json!(true), "{t}");
}

/// Both sensors with gaps, at different frames: each stream accounts for
/// its own ten slots, the camera frames of the lidar's gap have no sweep, the
/// sweep of the camera's gap has no frame, and every other instant pairs.
#[test]
fn gaps_on_both_sensors_at_different_frames_are_each_accounted_for() {
    let fx = gapped(&[7], &[3, 4], PACED_PERIOD_NS);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name", "mixed", "--track", "on", "--rate", "1.0", "--cap", "16",
        ],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert_all_invariants_ok(&out);
    let csv = evidence(&cwd, "mixed");
    assert_eq!(absent_seqs(&csv, "velo"), vec![3, 4]);
    assert_eq!(absent_seqs(&csv, "cam0"), vec![7]);
    assert_eq!(driver_rows(&csv, "velo").len(), FRAMES);
    assert_eq!(driver_rows(&csv, "cam0").len(), FRAMES);
    assert_frames_keep_their_numbers(&csv, PACED_PERIOD_NS);

    let s = summary(&cwd, "mixed");
    let t = &s["lidar"]["reduce"]["detect"]["track"];
    assert_eq!(t["source_gaps"]["no_sweep"], serde_json::json!([3, 4]));
    assert_eq!(t["source_gaps"]["no_frame"], serde_json::json!([7]));
    assert_eq!(
        t["source_gaps"]["no_frame_sweep_expired"],
        serde_json::json!([7]),
        "{t}"
    );
    assert_eq!(
        t["source_gaps"]["no_sweep_camera_reached_track"],
        serde_json::json!([3, 4]),
        "{t}"
    );
    assert!(
        out.contains("fusion sets for frames 3-4: none, no sweep in the lidar's source"),
        "{out}"
    );
    assert!(
        out.contains("fusion sets for frame 7: no camera frame in the source"),
        "{out}"
    );
    assert_eq!(t["cam_pair_join_ok"], serde_json::json!(true), "{t}");
    // No set named a camera frame of the lidar's gap: containment cannot,
    // and nothing asked for a stale window.
    assert_eq!(
        t["source_gaps"]["no_sweep_camera_used_stale"],
        serde_json::json!([])
    );
    let b = &t["sweeps_before_track"];
    assert_eq!(
        u64_of(b, "sweeps") + u64_of(b, "absent_in_source"),
        10,
        "{b}"
    );
    // Two WARN events, one per gap, each naming its sensor.
    let events = std::fs::read_to_string(cwd.path().join("runs/mixed/events.csv")).unwrap();
    let gap_events: Vec<&str> = events.lines().filter(|l| l.contains("SourceGap")).collect();
    assert_eq!(gap_events.len(), 2, "{events}");
    assert!(
        gap_events.iter().any(|l| l.contains("lidar frames 3-4")),
        "{events}"
    );
    assert!(
        gap_events.iter().any(|l| l.contains("camera frame 7")),
        "{events}"
    );
}

/// One frame, one number, on every edge and stage. Every row of the lidar's
/// chain -- the sweep, the voxels, the detections, the fusion's set or its
/// expiry, the answer -- carries the seq of the sweep whose instant it
/// carries, and every row of the camera's the seq of its frame. So after the
/// lidar's gap sweep 5 is 5 everywhere, and the fusion's expiry for the
/// camera's frame 7 is row 7, not the count of sets so far.
#[test]
fn every_edge_and_stage_carries_the_frame_s_own_number() {
    let fx = gapped(&[7], &[3, 4], PACED_PERIOD_NS);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name", "oneseq", "--track", "on", "--rate", "1.0", "--cap", "16",
        ],
    );
    assert_ok(&o);
    assert_all_invariants_ok(&stdout(&o));
    let csv = evidence(&cwd, "oneseq");
    let base = pipes_kitti::testing::FIXTURE_BASE_NS;
    let lidar = [
        "velo",
        "velo->cloud",
        "velo->reduce",
        "det",
        "det->cloud",
        "det->detect",
        "obj",
        "obj->sink",
        "obj->track",
        "track",
        "track->state",
        "state",
        "state->sink",
    ];
    let camera = ["cam0", "cam0->proc", "cam_det", "cam_det->track"];
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for r in &csv.rows {
        let edge = csv.col(r, "edge");
        let seq: i64 = csv.col(r, "seq").parse().expect("seq");
        let tov: i64 = csv.col(r, "tov_start_ns").parse().expect("tov");
        if tov == 0 {
            continue; // a frame absent in the source: no instant to compare
        }
        let frame = (tov - base) / PACED_PERIOD_NS;
        if let Some(e) = lidar.iter().chain(camera.iter()).find(|e| **e == edge) {
            assert_eq!(
                seq, frame,
                "{edge} carries frame {frame}'s instant as seq {seq}"
            );
            seen.insert(e);
        }
    }
    assert_eq!(
        seen.len(),
        lidar.len() + camera.len(),
        "not every edge was checked: {seen:?}"
    );
    // Sweep 5, the first after the lidar's gap, is 5 on every lidar edge.
    for edge in lidar {
        assert!(
            csv.rows
                .iter()
                .any(|r| csv.col(r, "edge") == edge && csv.col(r, "seq") == "5"),
            "no row for sweep 5 on {edge}"
        );
    }
    // The fusion's expiry for the camera's frame 7 is row 7.
    let expired: Vec<String> = csv
        .rows
        .iter()
        .filter(|r| {
            csv.col(r, "edge") == "track" && csv.col(r, "reason") == "pair_absent_in_source"
        })
        .map(|r| csv.col(r, "seq"))
        .collect();
    assert_eq!(expired, vec!["7".to_string()]);
    let s = summary(&cwd, "oneseq");
    assert_eq!(
        s["lidar"]["reduce"]["detect"]["track"]["expired_absent_in_source"],
        serde_json::json!([7])
    );
}

/// A drive with a sample for every frame on both sensors is what it was:
/// no gap lines, no absent rows, and the old spellings of every line.
#[test]
fn a_drive_with_no_gap_prints_what_it_always_has() {
    let fx = gapped(&[], &[], PERIOD_NS);
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "whole", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert_all_invariants_ok(&out);
    assert!(
        out.lines()
            .any(|l| l == "velo admitted=10 missing=0 n_sweeps=10"),
        "{out}"
    );
    assert!(
        out.lines()
            .any(|l| l == "INVARIANT driver=velo admitted=10 missing=0 n_sweeps=10 -> OK"),
        "{out}"
    );
    assert!(
        out.lines()
            .any(|l| l == "INVARIANT driver admitted=10 missing=0 n_frames=10 -> OK"),
        "{out}"
    );
    for absent in ["absent in source", "absent_in_source", "cam0 on_disk"] {
        assert!(
            !out.contains(absent),
            "{absent:?} on a drive with no gap:\n{out}"
        );
    }
    let csv = evidence(&cwd, "whole");
    assert!(absent_seqs(&csv, "velo").is_empty());
    assert!(absent_seqs(&csv, "cam0").is_empty());
    let events = std::fs::read_to_string(cwd.path().join("runs/whole/events.csv")).unwrap();
    assert!(!events.contains("SourceGap"), "{events}");
}

/// The data root the real-drive test reads: `PIPES_KITTI_ROOT`, or the
/// dataset beside the checkout, as the binary's default expects.
fn kitti_root() -> PathBuf {
    std::env::var("PIPES_KITTI_ROOT").map_or_else(
        |_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/kitti"),
        PathBuf::from,
    )
}

/// KITTI's own gap: drive 0009's lidar has no sweeps for frames 177-180.
/// The run opens it, replays 443 sweeps and records the four absent frames,
/// and accounts for all 447 frame slots. Fails loudly, naming the path, when
/// the drive is not on disk and the test is asked to run.
#[test]
#[ignore = "needs drive_0009 at PIPES_KITTI_ROOT"]
fn drive_0009_replays_443_sweeps_and_records_frames_177_to_180_as_absent() {
    const DRIVE: &str = "2011_09_26_drive_0009_sync";
    let root = kitti_root();
    let dir = root.join("2011_09_26").join(DRIVE).join("velodyne_points");
    assert!(
        dir.is_dir(),
        "no velodyne data at {} -- set PIPES_KITTI_ROOT",
        dir.display()
    );
    let cwd = TempDir::new().unwrap();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_pipes"))
        .current_dir(cwd.path())
        .env_remove("PIPES_KITTI_ROOT")
        .args(["run", "--kitti-root"])
        .arg(&root)
        .args([
            "--drive",
            DRIVE,
            "--name",
            "d0009",
            "--rate",
            "inf",
            "--rerun",
            "null",
            "--dashboard",
            "off",
            "--track",
            "on",
            "--detector",
            "off",
        ])
        .output()
        .expect("spawn pipes");
    assert_ok(&o);
    let out = stdout(&o);
    assert_all_invariants_ok(&out);
    let csv = evidence(&cwd, "d0009");
    let velo = driver_rows(&csv, "velo");
    assert_eq!(velo.len(), 447, "one driver row per frame slot");
    assert_eq!(
        velo.iter().filter(|r| r.1 == "Delivered").count(),
        443,
        "unpaced, so every sweep is admitted"
    );
    assert_eq!(absent_seqs(&csv, "velo"), vec![177, 178, 179, 180]);
    assert!(absent_seqs(&csv, "cam0").is_empty());
    assert!(
        out.lines().any(|l| l
            == "velo admitted=443 missing=4 n_sweeps=443 frames=447 absent_in_source=4 (frames 177-180: no sweep in the source)"),
        "{out}"
    );
    let s = summary(&cwd, "d0009");
    assert_eq!(
        s["lidar"]["absent_in_source"],
        serde_json::json!([177, 178, 179, 180])
    );
    let b = &s["lidar"]["reduce"]["detect"]["track"]["sweeps_before_track"];
    assert_eq!(
        (u64_of(b, "sweeps"), u64_of(b, "absent_in_source")),
        (443, 4),
        "{b}"
    );
    assert!(
        line_starting(&out, "velo 2011_09_26_drive_0009_sync: ").contains(
            "frames 177-180 (4 frames, 413.9 ms with no measurement between frame 176 and frame 181)"
        ),
        "{out}"
    );
}
