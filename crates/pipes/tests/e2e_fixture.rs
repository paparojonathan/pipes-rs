//! The whole pipeline, end to end, with no dataset.
//!
//! `crates/pipes` is binary-only, so there is nothing to `use pipes::…`; the
//! pipeline is tested by running the real binary through
//! `env!("CARGO_BIN_EXE_pipes")`. Every run gets its own `TempDir` as its CWD,
//! so `runs/` lands there and the tests are safe in parallel.
//!
//! The binary's exit code already carries the H.3 invariant result (0 = every
//! invariant OK, 2 = a FAIL), so `status.success()` *is* an assertion about
//! the invariants.

use pipes_kitti::testing::{FixtureDrive, FIXTURE_DATE, FIXTURE_DRIVE, FIXTURE_H, FIXTURE_W};
use tempfile::TempDir;

mod common;
use common::{assert_line, assert_ok, evidence, pipes, run_json, stdout, summary, EVIDENCE_HEADER};

#[test]
fn e2e_unpaced_no_drops_all_invariants_ok() {
    let fx = FixtureDrive::new(16).unwrap();
    let cwd = TempDir::new().unwrap();
    // --cap 16 with 16 frames: the queue holds every frame the driver can ever
    // admit, so PushOutcome is Accepted every time regardless of scheduling.
    // That is what makes the exact counts below legitimate rather than lucky.
    let o = pipes(&cwd, &fx, &["--name", "e2e", "--cap", "16"]);
    assert_ok(&o);
    assert!(
        stdout(&o)
            .lines()
            .any(|l| l.starts_with("driver admitted=16 missing=0 wall_s=")),
        "--- stdout ---\n{}",
        stdout(&o)
    );
    assert_line(&o, "admitted = 16");
    assert_line(
        &o,
        "edge=cam0->proc delivered=16 dropped=0 admitted=16 storage_mismatch=0",
    );
    assert_line(&o, "storage_id equal at 3 stages: OK");
    assert_line(&o, "proc bytes/frame = 32");
    assert_line(&o, "rerun bytes/frame = 0");
    assert_line(
        &o,
        "INVARIANT driver admitted=16 missing=0 n_frames=16 -> OK",
    );
    assert_line(
        &o,
        "INVARIANT edge=cam0->proc delivered=16 dropped=0 admitted=16 -> OK",
    );
    assert_line(
        &o,
        "INVARIANT rows edge=cam0->proc rows=16 admitted=16 -> OK",
    );
    assert_line(
        &o,
        "INVARIANT rows edge=cam0->rerun rows=16 admitted=16 -> OK",
    );
    assert_line(&o, "evidence rows = 48 lost = 0 recorder_degraded = false");
    assert!(
        !stdout(&o).contains("-> FAIL"),
        "an invariant FAILed:\n{}",
        stdout(&o)
    );
}

#[test]
fn e2e_reuse_output_allocates_zero_bytes_per_frame() {
    let fx = FixtureDrive::new(16).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "reuse", "--cap", "16", "--reuse-output"],
    );
    assert_ok(&o);
    // The claim of the whole --reuse-output path: the per-frame output
    // allocation goes away entirely, it does not merely shrink.
    assert_line(&o, "proc bytes/frame = 0");
    assert_eq!(
        stdout(&o)
            .lines()
            .filter(|l| *l == "proc reuse buffer = 32 bytes")
            .count(),
        1,
        "the buffer must be set up exactly once, not per frame"
    );
    assert_line(
        &o,
        "edge=cam0->proc delivered=16 dropped=0 admitted=16 storage_mismatch=0",
    );
}

/// `proc` computes one grayscale frame and reports nothing else: the columns
/// the old `edges` / `track` workloads filled are gone from the header, so a
/// reader of an old run directory cannot mistake this one for it.
#[test]
fn e2e_default_run_is_grayscale_and_writes_the_26_columns() {
    let fx = FixtureDrive::new(16).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "plain", "--cap", "16"]);
    assert_ok(&o);
    assert_line(&o, "proc bytes/frame = 32");
    let csv = evidence(&cwd, "plain");
    assert_eq!(csv.header.len(), 26, "{:?}", csv.header);
    for gone in ["motion_mad", "tracks_active", "track_switches"] {
        assert!(
            !csv.header.iter().any(|h| h == gone),
            "{gone} is still a column"
        );
    }
    let s = summary(&cwd, "plain");
    for gone in ["proc_work", "motion", "tracks"] {
        assert!(s.get(gone).is_none(), "summary.json still carries `{gone}`");
    }
}

#[test]
fn e2e_evidence_csv_header_and_rows() {
    let fx = FixtureDrive::new(16).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "e2e", "--cap", "16"]);
    assert_ok(&o);

    let path = cwd.path().join("runs").join("e2e").join("evidence.csv");
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.lines().next().unwrap(), EVIDENCE_HEADER);
    assert_eq!(text.lines().count() - 1, 48, "48 data rows expected");

    let csv = evidence(&cwd, "e2e");
    assert_eq!(csv.rows.len(), 48);
    // One row per admitted sample per edge, whatever happened to it.
    for edge in ["cam0", "cam0->proc", "cam0->rerun"] {
        assert_eq!(csv.with("edge", edge).len(), 16, "{edge} rows");
    }
    // The driver and the cap-16 proc edge are exact: nothing can be dropped.
    for (edge, stage) in [("cam0", "driver"), ("cam0->proc", "proc")] {
        let n = csv
            .rows
            .iter()
            .filter(|r| csv.col(r, "edge") == edge && csv.col(r, "stage") == stage)
            .count();
        assert_eq!(n, 16, "{edge}/{stage} rows");
    }
    // cam0->rerun is a hard-coded cap-1 DropOldest edge, so how many frames
    // the consumer got is scheduling-dependent and asserting a count there
    // would be asserting the scheduler. Only the conservation identity holds
    // unconditionally: every row is either the consumer's or the pusher's.
    let rerun_rows = csv.with("edge", "cam0->rerun");
    let delivered = rerun_rows
        .iter()
        .filter(|r| csv.col(r, "stage") == "rerun")
        .count();
    let dropped = rerun_rows
        .iter()
        .filter(|r| csv.col(r, "stage") == "admission")
        .count();
    assert_eq!(delivered + dropped, 16, "a cam0->rerun row from nowhere");
    assert!(delivered >= 1, "the rerun consumer received nothing at all");

    for r in csv.with("stage", "driver") {
        assert_eq!(csv.col(r, "outcome"), "Delivered");
    }
    assert!(
        !text.contains("Missing"),
        "an unpaced run has no deadlines, so it cannot skip one"
    );
}

#[test]
fn e2e_summary_json_invariants_ok() {
    let fx = FixtureDrive::new(16).unwrap();
    let cwd = TempDir::new().unwrap();
    assert_ok(&pipes(&cwd, &fx, &["--name", "e2e", "--cap", "16"]));
    let s = summary(&cwd, "e2e");

    assert_eq!(s["invariants_ok"], serde_json::json!(true));
    assert_eq!(s["storage_id_equal_all_stages"], serde_json::json!(true));
    assert_eq!(s["n_frames"], serde_json::json!(16));
    assert_eq!(s["admitted"], serde_json::json!(16));
    assert_eq!(s["missing"], serde_json::json!(0));
    assert_eq!(s["evidence_lost"], serde_json::json!(0));
    assert_eq!(s["recorder_degraded"], serde_json::json!(false));
    assert_eq!(s["policy"], serde_json::json!("drop-oldest"));
    assert_eq!(s["rerun_mode"], serde_json::json!("null"));
    assert_eq!(
        s["bytes_alloc_semantics"],
        serde_json::json!(
            "bytes requested from the allocator on this thread (allocations + realloc growth), not resident memory"
        )
    );

    let stages = s["stages"].as_array().expect("stages array");
    let edges: Vec<&str> = stages.iter().filter_map(|st| st["edge"].as_str()).collect();
    assert_eq!(edges, vec!["cam0", "cam0->proc", "cam0->rerun"]);
    let proc = &stages[1];
    assert_eq!(proc["delivered"], serde_json::json!(16));
    assert_eq!(proc["dropped_oldest"], serde_json::json!(0));
    assert_eq!(
        proc["bytes_alloc_per_frame_mean"],
        serde_json::json!(32.0),
        "8x4 grayscale is one 32-byte output per frame"
    );
}

#[test]
fn e2e_rerun_off_is_two_stages() {
    let fx = FixtureDrive::new(16).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "off", "--cap", "16", "--rerun", "off"],
    );
    assert_ok(&o);
    assert_line(&o, "storage_id equal at 2 stages (rerun off): OK");
    assert_line(&o, "rerun bytes/frame = n/a (off)");
    assert_line(&o, "evidence rows = 32 lost = 0 recorder_degraded = false");
    let csv = evidence(&cwd, "off");
    assert_eq!(csv.rows.len(), 32);
    assert!(
        csv.with("edge", "cam0->rerun").is_empty(),
        "there is no rerun edge with --rerun off"
    );
    assert_eq!(csv.with("edge", "cam0->proc").len(), 16);
}

#[test]
fn e2e_rrd_file_is_written() {
    let fx = FixtureDrive::new(16).unwrap();
    let cwd = TempDir::new().unwrap();
    // No viewer and no network: RecordingStreamBuilder::save writes a local file.
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "rrd", "--cap", "16", "--rerun", "rrd"],
    );
    assert_ok(&o);
    let rrd = cwd.path().join("runs").join("rrd").join("cam0.rrd");
    let len = std::fs::metadata(&rrd)
        .unwrap_or_else(|e| panic!("{}: {e}", rrd.display()))
        .len();
    assert!(len > 0, "{} is empty", rrd.display());
    assert_line(&o, "storage_id equal at 3 stages: OK");
    // And the run says where it put things, as full paths, so a reader of
    // the terminal does not have to know where the run was started from.
    assert_line(
        &o,
        &format!("run dir: {}", cwd.path().join("runs").join("rrd").display()),
    );
    assert_line(&o, &format!("rerun: {}", rrd.display()));
}

/// The plain test run has no recording; the last line says so rather than
/// naming a file that is not there.
#[test]
fn e2e_a_run_without_a_recording_says_so() {
    let fx = FixtureDrive::new(4).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "none", "--cap", "16"]);
    assert_ok(&o);
    assert_line(&o, "rerun: nothing recorded (--rerun null)");
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "dash", "--cap", "16", "--dashboard", "on"],
    );
    assert_ok(&o);
    assert_line(
        &o,
        &format!(
            "rerun: {}",
            cwd.path()
                .join("runs")
                .join("dash")
                .join("dashboard.rrd")
                .display()
        ),
    );
}

#[test]
fn e2e_paced_run_records_due_and_age() {
    // 8 frames 10 ms apart at rate 10 => 1 ms deadlines.
    let fx = FixtureDrive::with_period(8, 10_000_000).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "paced", "--cap", "16", "--rate", "10"],
    );
    assert_ok(&o);

    // `missing == 0` is deliberately NOT asserted: a runner that stalls past a
    // deadline is the condition Missing{deadline_skipped} exists to record.
    // Conservation holds unconditionally, so that is what is pinned.
    let inv = stdout(&o)
        .lines()
        .find(|l| l.starts_with("INVARIANT driver admitted="))
        .unwrap_or_else(|| panic!("no driver INVARIANT line:\n{}", stdout(&o)))
        .to_string();
    assert!(inv.ends_with("n_frames=8 -> OK"), "{inv}");
    let num = |key: &str| -> u64 {
        inv.split_whitespace()
            .find_map(|t| t.strip_prefix(key))
            .unwrap_or_else(|| panic!("no {key} in {inv}"))
            .parse()
            .unwrap()
    };
    let (a, m) = (num("admitted="), num("missing="));
    assert_eq!(a + m, 8, "{inv}");
    assert!(a >= 1, "a paced run at rate 10 admitted nothing: {inv}");

    let csv = evidence(&cwd, "paced");
    let driver_admitted: Vec<_> = csv
        .rows
        .iter()
        .filter(|r| csv.col(r, "stage") == "driver" && csv.col(r, "outcome") == "Delivered")
        .collect();
    assert_eq!(driver_admitted.len() as u64, a);
    for r in &driver_admitted {
        assert!(
            !csv.col(r, "due_ns").is_empty(),
            "a paced frame with no deadline: seq {}",
            csv.col(r, "seq")
        );
    }
    let proc_delivered: Vec<_> = csv
        .rows
        .iter()
        .filter(|r| csv.col(r, "edge") == "cam0->proc" && csv.col(r, "outcome") == "Delivered")
        .collect();
    assert!(!proc_delivered.is_empty(), "nothing reached proc");
    for r in &proc_delivered {
        assert!(!csv.col(r, "queue_wait_ns").is_empty());
        assert!(
            !csv.col(r, "measurement_age_ns").is_empty(),
            "measurement age needs a deadline; seq {}",
            csv.col(r, "seq")
        );
    }

    let s = summary(&cwd, "paced");
    assert!(
        s["pacing_us"]["p50"].is_number(),
        "pacing_us.p50 is {} -- a paced run has a pacing error",
        s["pacing_us"]["p50"]
    );
    let steady = s["stages"][1]["steady_rows"].as_u64().unwrap();
    assert!(steady >= 1, "no steady rows in a {a}-frame paced run");
}

/// Provenance. `run.json` recorded `drive` and `n_frames` and neither the
/// capture date nor the resolution, which was survivable while exactly one
/// scene existed. With several it is not: KITTI's frame size is a property of
/// the *date* (1242x375, 1224x370, 1238x374, 1226x370, 1241x376), so a run
/// directory that does not say which date it came from cannot be read back —
/// `proc_bytes_per_frame` is w*h and means nothing without w and h.
#[test]
fn run_json_records_the_scene_and_its_resolution() {
    let fx = FixtureDrive::new(8).unwrap();
    let cwd = TempDir::new().unwrap();
    assert_ok(&pipes(&cwd, &fx, &["--name", "prov", "--cap", "8"]));

    let j = run_json(&cwd, "prov");
    assert_eq!(j["date"], serde_json::json!(FIXTURE_DATE));
    assert_eq!(j["drive"], serde_json::json!(FIXTURE_DRIVE));
    assert_eq!(j["width"], serde_json::json!(FIXTURE_W));
    assert_eq!(j["height"], serde_json::json!(FIXTURE_H));

    // The fields that were already there keep their names and their values:
    // 43 committed run directories are read by the same key set.
    assert_eq!(j["n_frames"], serde_json::json!(8));
    assert_eq!(j["rerun_mode"], serde_json::json!("null"));
    assert!(j["clock"].is_object(), "clock: {}", j["clock"]);
    assert!(j["git_sha"].is_string());
    assert!(j["args"].is_array());
    assert!(j["spin_window_ms"].is_number());
    assert_eq!(j["rate"], serde_json::json!("inf"));

    // And whether the host could throttle it: on Windows the run opts out,
    // so its timings do not depend on which window had the focus.
    let want = if cfg!(windows) {
        "off"
    } else {
        "not applicable"
    };
    assert_eq!(j["power_throttling"], serde_json::json!(want));
}
