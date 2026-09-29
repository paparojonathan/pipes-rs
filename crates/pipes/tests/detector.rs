//! The `--detector` switch, end to end.
//!
//! What CI can check without the model is the refusal: `--detector on` is a
//! request for the camera to contribute what it SAW, and a run that cannot
//! load the frozen model must stop -- naming the download -- rather than fuse
//! with the bare frame reference and report a camera that looked as one that
//! did. The switch is also scoped: it reaches only a run that fuses, so a run
//! that does not must not need the model.
//!
//! What needs the model and drive_0005 is gated on the terms the other
//! real-drive tests are (`#[ignore]`, run by hand):
//!
//! ```text
//! cargo test --release -p pipes --test detector -- --ignored --nocapture
//! ```
//!
//! It runs the real binary twice over the whole drive and pins that the
//! detector fed the fusion, that the evidence balances, and that the two runs'
//! `CAM_DET` batches are byte-identical over the frames both delivered; that a
//! stale camera changes the camera lanes and nothing else; and that slowing
//! the detector with `--consumer-delay-ms` expires exactly the sweeps whose
//! frames it lost.

use std::path::{Path, PathBuf};
use std::process::Command;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::Int64Type;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use pipes_kitti::camdet::{FETCH_SCRIPT, MODEL_FILE, MODEL_SHA256};
use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{
    assert_ok, dropped_on, evidence, line_with, pipes, run_dir, stderr, stdout, summary, u64_of,
};

const N: usize = 8;
const PERIOD_NS: i64 = 5_000_000;

/// A lidar fixture with its calibration: everything `--track on` needs, and
/// no model -- the test's CWD is a fresh temp dir with no `models/` in it.
fn fused_fx() -> FixtureDrive {
    let fx = FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    fx
}

/// The refusal, and the message that ends at the next step.
#[test]
fn a_missing_model_stops_the_run_and_names_the_download() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "nomodel", "--track", "on", "--detector", "on"],
    );
    assert!(
        !o.status.success(),
        "a run with `--detector on` and no model SUCCEEDED:\n{}",
        stdout(&o)
    );
    let err = stderr(&o);
    // The script as a PowerShell user types it, and the way out.
    let script = FETCH_SCRIPT.replace('/', "\\");
    assert!(
        err.contains(&script),
        "the error does not name {script}:\n{err}"
    );
    assert!(err.contains("--detector off"), "{err}");
    assert!(
        err.contains(MODEL_FILE),
        "the error does not name the file:\n{err}"
    );
    // Refused before anything was written: no run directory at all, so no
    // half-run can later be read as a run.
    assert!(
        !run_dir(&cwd, "nomodel").exists(),
        "the refused run left a directory behind"
    );
    // Positive control: the same fixture and flags with the detector off run.
    let o2 = pipes(
        &cwd,
        &fx,
        &["--name", "ref", "--track", "on", "--detector", "off"],
    );
    assert_ok(&o2);
    assert!(stdout(&o2).contains("fused cam_seq = a frame the camera really admitted"));
}

/// A file that is there but is not the pinned model is refused by its hash,
/// with the same next step.
#[test]
fn a_model_that_is_not_the_pinned_one_is_refused() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let path = cwd.path().join(MODEL_FILE);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"not the model").unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "wrong", "--track", "on", "--detector", "on"],
    );
    assert!(!o.status.success(), "a wrong model file was accepted");
    let err = stderr(&o);
    assert!(err.contains("sha256"), "{err}");
    assert!(
        err.contains(MODEL_SHA256),
        "the error does not say what was expected:\n{err}"
    );
    assert!(err.contains(&FETCH_SCRIPT.replace('/', "\\")), "{err}");
}

/// `--detector on` reaches only a run that fuses: without `--track on` it
/// changes nothing, so it cannot need the model. (The positive control is the
/// test above: with `--track on` the same absence is fatal.)
#[test]
fn the_detector_needs_the_model_only_where_there_is_a_fusion() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "notrack", "--track", "off", "--detector", "on"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert!(!out.contains("cam0->camdet"), "{out}");
    assert!(summary(&cwd, "notrack")["camera_detector"].is_null());
    assert!(
        !run_dir(&cwd, "notrack").join("cam_det.arrows").exists(),
        "a run without the detector wrote detector batches"
    );
}

/// `--cap` and `--consumer-delay-ms` act on the camera stage whose output
/// reaches the answer, and where no detector can run -- here `--track off`,
/// with `--detector on` asked for -- that stage is `proc`. The run says which
/// queue the knobs set, and the positive control is that `proc`'s queue then
/// drops. Where the detector does run the knobs reach `cam0->camdet`: pinned
/// without the model by `run::tests`, and end to end by the gated
/// `slowing_the_detector_expires_exactly_the_frames_it_lost` below.
#[test]
fn where_no_detector_runs_the_camera_knobs_reach_proc() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "knobs",
            "--track",
            "off",
            "--detector",
            "on",
            "--cap",
            "1",
            "--consumer-delay-ms",
            "20",
            "--rate",
            "1",
        ],
    );
    assert_ok(&o);
    let out = stdout(&o);
    let run = line_with(&out, "run knobs ");
    for field in ["cap=1 ", "delay_ms=20 ", "camera_queue=cam0->proc "] {
        assert!(run.contains(field), "no {field:?} in {run:?}");
    }
    let s = summary(&cwd, "knobs");
    assert_eq!(s["camera_queue"], serde_json::json!("cam0->proc"), "{s}");
    assert_eq!(s["cap"], serde_json::json!(1), "{s}");
    assert_eq!(s["delay_ms"], serde_json::json!(20), "{s}");
    // 8 frames 5 ms apart into a 1-deep queue whose consumer sleeps 20 ms a
    // frame: the knobs reached proc only if it lost frames.
    assert!(
        dropped_on(&s, "cam0->proc") > 0,
        "proc dropped nothing, so the knobs did not reach it: {s}"
    );
    assert!(!out.contains("cam0->camdet"), "{out}");
}

// ---- gated: the model and drive_0005 -------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn kitti_root() -> PathBuf {
    std::env::var("PIPES_KITTI_ROOT")
        .map_or_else(|_| repo_root().join("../data/kitti"), PathBuf::from)
}

/// One real run in `cwd`, which holds a copy of the model.
fn real_run(cwd: &Path, name: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_pipes"))
        .current_dir(cwd)
        .args(["run", "--kitti-root"])
        .arg(kitti_root())
        .args(["--name", name, "--rerun", "null", "--dashboard", "off"])
        .output()
        .expect("spawn pipes")
}

/// Every batch in a run's `cam_det.arrows`, by frame, as the bytes one batch
/// re-encodes to.
fn batches_by_frame(dir: &Path) -> std::collections::BTreeMap<i64, Vec<u8>> {
    let f = std::fs::File::open(dir.join("cam_det.arrows")).expect("cam_det.arrows");
    let reader = StreamReader::try_new(f, None).expect("an Arrow IPC stream");
    let mut out = std::collections::BTreeMap::new();
    for b in reader {
        let b: RecordBatch = b.expect("a batch");
        let seq = b
            .column_by_name("frame_seq")
            .expect("frame_seq")
            .as_primitive::<Int64Type>()
            .value(0);
        let mut bytes = Vec::new();
        let mut w = StreamWriter::try_new(&mut bytes, &b.schema()).unwrap();
        w.write(&b).unwrap();
        w.finish().unwrap();
        drop(w);
        assert!(out.insert(seq, bytes).is_none(), "frame {seq} twice");
    }
    out
}

#[test]
#[ignore = "needs drive_0005 and models/yolox_nano.onnx (scripts/fetch_model.ps1)"]
fn two_real_runs_feed_the_fusion_the_same_detections() {
    let cwd = TempDir::new().unwrap();
    let model = cwd.path().join(MODEL_FILE);
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::copy(repo_root().join(MODEL_FILE), &model)
        .unwrap_or_else(|e| panic!("{}: {e}", repo_root().join(MODEL_FILE).display()));

    let mut runs = Vec::new();
    for name in ["det-a", "det-b"] {
        let o = real_run(cwd.path(), name);
        assert!(
            o.status.success(),
            "{name}: exit {:?}\n{}\n{}",
            o.status.code(),
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        let out = String::from_utf8_lossy(&o.stdout).into_owned();
        for line in [
            "INVARIANT stage=camdet ",
            "INVARIANT edge=cam0->camdet ",
            "INVARIANT rows edge=cam0->camdet ",
            "INVARIANT edge=cam_det->track ",
        ] {
            let l = out
                .lines()
                .find(|l| l.starts_with(line))
                .unwrap_or_else(|| panic!("{name}: no {line:?} line:\n{out}"));
            assert!(l.ends_with("-> OK"), "{name}: {l}");
        }
        assert!(
            out.contains("track read camdet's buffer: OK"),
            "{name}: the fusion did not read the detector's batches in place:\n{out}"
        );
        let s = summary(&cwd, name);
        let c = s["camera_detector"].clone();
        assert_eq!(c["model_sha256"], serde_json::json!(MODEL_SHA256), "{c}");
        assert_eq!(c["input_storage_id_equal"], serde_json::json!(true), "{c}");
        let produced = u64_of(&c, "produced");
        assert!(
            produced > 100,
            "{name}: the detector ran on {produced} frames: {c}"
        );
        let t = s["lidar"]["reduce"]["detect"]["track"].clone();
        assert!(
            u64_of(&t, "completed") > 0,
            "{name}: nothing was fused: {t}"
        );
        let batches = batches_by_frame(&run_dir(&cwd, name));
        assert_eq!(
            batches.len() as u64,
            produced,
            "{name}: file against summary"
        );
        runs.push(batches);
    }

    // Frozen: over the frames both runs delivered, the same bytes.
    let (a, b) = (&runs[0], &runs[1]);
    let common: Vec<&i64> = a.keys().filter(|k| b.contains_key(*k)).collect();
    assert!(common.len() > 100, "the runs share {} frames", common.len());
    for k in &common {
        assert_eq!(a[*k], b[*k], "frame {k}: the two runs' detections differ");
    }
    // Positive control: the comparison tells two frames apart.
    let (first, last) = (common[0], common[common.len() - 1]);
    assert_ne!(
        a[first], a[last],
        "every frame encodes the same, so equality says nothing"
    );
    println!("{} frames compared, byte-identical", common.len());
}

/// The fusion's sets of one run, by sweep: `(cam_seq, pair_age_ns,
/// pair_outcome, the 26-lane tracks)`.
#[allow(clippy::type_complexity)]
fn fused_by_sweep(
    dir: &Path,
) -> std::collections::BTreeMap<
    i64,
    (
        i64,
        i64,
        String,
        Vec<[f32; pipes_kitti::track::TRACK_LANES]>,
    ),
> {
    use pipes_kitti::track::{cam_seq, pair_age_ns, pair_outcome, track_rows, track_sweep_seq};
    let f = std::fs::File::open(dir.join("fused.arrows")).expect("fused.arrows");
    let mut out = std::collections::BTreeMap::new();
    for b in StreamReader::try_new(f, None).expect("an Arrow IPC stream") {
        let b: RecordBatch = b.expect("a batch");
        let sweep = track_sweep_seq(&b).expect("source_sweep_seq");
        let set = (
            cam_seq(&b).expect("cam_seq"),
            pair_age_ns(&b).expect("pair_age_ns"),
            pair_outcome(&b).expect("pair_outcome").to_string(),
            track_rows(&b).to_vec(),
        );
        assert!(out.insert(sweep, set).is_none(), "sweep {sweep} twice");
    }
    out
}

/// **The stale-camera demonstration, as a test.** Two real runs that differ in
/// one flag: the control waits for the camera (`--pair-stale-ms 500`), the
/// treatment does not (`--pair-stale-ms 500 --pair-wait-ms 0`), so the
/// detector has not finished the sweep's own frame when the fusion needs it
/// and the declared window hands it the previous one. Both runs slow the
/// detector by 50 ms a frame (`--consumer-delay-ms 50`): a detector faster
/// than the lidar chain would have the frame ready in time, and the treatment
/// would never go stale.
///
/// Pinned: the lidar half of every sweep is byte-identical between the runs,
/// so any difference is the camera's; the treatment's stale sets name an
/// older frame, at a negative pair age, with `reason = stale` on their rows;
/// and on the sweeps that were stale, the fusion confirmed FEWER tracks than
/// the control did from the fresh frame -- the corruption, measured. The size
/// of the drop is the run's to report, not this test's: it asserts the
/// direction and that the evidence names the cause.
#[test]
#[ignore = "needs drive_0005 and models/yolox_nano.onnx (scripts/fetch_model.ps1)"]
fn a_stale_camera_changes_the_camera_lanes_and_nothing_else() {
    let cwd = TempDir::new().unwrap();
    let model = cwd.path().join(MODEL_FILE);
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::copy(repo_root().join(MODEL_FILE), &model)
        .unwrap_or_else(|e| panic!("{}: {e}", repo_root().join(MODEL_FILE).display()));
    for (name, extra) in [
        (
            "healthy",
            vec!["--consumer-delay-ms", "50", "--pair-stale-ms", "500"],
        ),
        (
            "stale",
            vec![
                "--consumer-delay-ms",
                "50",
                "--pair-stale-ms",
                "500",
                "--pair-wait-ms",
                "0",
            ],
        ),
    ] {
        let o = Command::new(env!("CARGO_BIN_EXE_pipes"))
            .current_dir(cwd.path())
            .args(["run", "--kitti-root"])
            .arg(kitti_root())
            .args(["--name", name, "--rerun", "null", "--dashboard", "off"])
            .args(&extra)
            .output()
            .expect("spawn pipes");
        assert!(
            o.status.success(),
            "{name}: exit {:?}\n{}\n{}",
            o.status.code(),
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        let out = String::from_utf8_lossy(&o.stdout).into_owned();
        let inv = out
            .lines()
            .find(|l| l.starts_with("INVARIANT fusion "))
            .unwrap_or_else(|| panic!("{name}: no INVARIANT fusion line:\n{out}"));
        assert!(inv.ends_with("-> OK"), "{name}: {inv}");
    }
    let (a, b) = (
        fused_by_sweep(&run_dir(&cwd, "healthy")),
        fused_by_sweep(&run_dir(&cwd, "stale")),
    );
    let fused = |rows: &[[f32; pipes_kitti::track::TRACK_LANES]]| {
        rows.iter()
            .filter(|r| {
                pipes_kitti::track::track_population(r) == pipes_kitti::fuse::POPULATION_FUSED
            })
            .count()
    };
    let (mut common, mut stale, mut fused_a, mut fused_b) = (0, 0, 0, 0);
    for (sweep, (cs, age, outcome, rows)) in &b {
        let Some((_, _, outcome_a, rows_a)) = a.get(sweep) else {
            continue;
        };
        common += 1;
        // The lidar half: lanes 0-20 of every track, bit for bit.
        assert_eq!(
            rows.len(),
            rows_a.len(),
            "sweep {sweep}: a different number of tracks"
        );
        for (x, y) in rows.iter().zip(rows_a) {
            let (x, y) = (&x[..21], &y[..21]);
            assert!(
                x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
                "sweep {sweep}: the lidar lanes differ, so the camera is not the only difference"
            );
        }
        if outcome == "stale" && outcome_a == "paired" {
            stale += 1;
            assert!(
                *age < 0,
                "sweep {sweep}: a stale set with pair age {age} ns"
            );
            assert!(*cs < *sweep, "sweep {sweep}: a stale set named frame {cs}");
            fused_a += fused(rows_a);
            fused_b += fused(rows);
        }
    }
    assert!(common > 100, "the runs share {common} sweeps");
    assert!(
        stale > 50,
        "only {stale} sweeps were stale, so the comparison is thin"
    );
    assert!(
        fused_b < fused_a,
        "on {stale} stale sweeps the fusion confirmed {fused_b} tracks against {fused_a} from the fresh frames: the stale camera changed nothing"
    );
    // The evidence names the cause without the payload: `reason = stale` on
    // exactly the treatment's degraded sets.
    let ev = std::fs::read_to_string(run_dir(&cwd, "stale").join("evidence.csv")).unwrap();
    let stale_rows = ev
        .lines()
        .filter(|l| l.contains(",track,track,") && l.contains(",Delivered,stale,"))
        .count();
    let degraded = b.values().filter(|s| s.2 == "stale").count();
    assert_eq!(
        stale_rows, degraded,
        "evidence.csv does not name every stale set"
    );
    println!(
        "{common} sweeps compared, lidar lanes identical; on {stale} stale sweeps: fused {fused_b} (stale) against {fused_a} (fresh)"
    );
}

/// **"Slow the camera, and the answer loses exactly those frames"**, with the
/// detector on: `--consumer-delay-ms 100 --cap 1 --pair-wait-ms 300` against
/// a plain run.
///
/// Pinned: the knobs reached `cam0->camdet` -- the detector's queue dropped
/// frames, `proc` did not run beside it, and the delay is inside the
/// detector's measured service time; every frame the detector's queue dropped
/// is a sweep the fusion expired as `pair_dropped`, and every expired set is
/// one of those, joined on sensor instants, so the check does not rest on
/// the frame numbering; no sweep was lost on the lidar side; and the
/// lidar lanes of every sweep the slowed run fused are bit for bit the plain
/// run's, so the camera is the only difference.
///
/// At 100 ms the detector spends up to about two camera periods a frame, and
/// the fusion's waits stay short enough for the lidar chain. At 200 ms,
/// nearly three, one run of four had the fusion fall behind and `obj->track`
/// evict 11 sweeps -- a loss on the lidar side, which this test would report;
/// a loaded host can push 100 ms toward that too.
#[test]
#[ignore = "needs drive_0005 and models/yolox_nano.onnx (scripts/fetch_model.ps1)"]
fn slowing_the_detector_expires_exactly_the_frames_it_lost() {
    let cwd = TempDir::new().unwrap();
    let model = cwd.path().join(MODEL_FILE);
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::copy(repo_root().join(MODEL_FILE), &model)
        .unwrap_or_else(|e| panic!("{}: {e}", repo_root().join(MODEL_FILE).display()));
    let mut outs = std::collections::BTreeMap::new();
    for (name, extra) in [
        ("plain", vec![]),
        (
            "slow",
            vec![
                "--consumer-delay-ms",
                "100",
                "--cap",
                "1",
                "--pair-wait-ms",
                "300",
            ],
        ),
    ] {
        let o = Command::new(env!("CARGO_BIN_EXE_pipes"))
            .current_dir(cwd.path())
            .args(["run", "--kitti-root"])
            .arg(kitti_root())
            .args(["--name", name, "--rerun", "null", "--dashboard", "off"])
            .args(&extra)
            .output()
            .expect("spawn pipes");
        // Exit 0 also says every INVARIANT line was OK: a FAIL exits 2.
        assert!(
            o.status.success(),
            "{name}: exit {:?}\n{}\n{}",
            o.status.code(),
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        outs.insert(name, String::from_utf8_lossy(&o.stdout).into_owned());
    }

    // The knobs landed on the detector's queue, and proc did not run.
    let run = line_with(&outs["slow"], "run slow ");
    for field in ["cap=1 ", "delay_ms=100 ", "camera_queue=cam0->camdet "] {
        assert!(run.contains(field), "no {field:?} in {run:?}");
    }
    let s = summary(&cwd, "slow");
    assert_eq!(s["camera_queue"], serde_json::json!("cam0->camdet"), "{s}");
    let lost = dropped_on(&s, "cam0->camdet");
    assert!(lost > 0, "slowing the detector cost it no frames: {s}");
    assert!(
        !s["stages"]
            .as_array()
            .expect("stages array")
            .iter()
            .any(|st| st["edge"] == serde_json::json!("cam0->proc")),
        "proc ran beside the detector: {s}"
    );
    let service_p50 = s["camera_detector"]["service_us"]["p50"]
        .as_i64()
        .expect("service_us.p50");
    assert!(
        service_p50 >= 100_000,
        "the detector's service p50 is {service_p50} us, so the 100 ms delay is not inside its window"
    );
    assert_eq!(
        dropped_on(&s, "obj->track"),
        0,
        "the fusion lost sweeps on the lidar side, so the answer lost more than the camera did"
    );
    let t = s["lidar"]["reduce"]["detect"]["track"].clone();
    assert_eq!(
        (u64_of(&t, "expired"), u64_of(&t, "expired_dropped")),
        (lost, lost),
        "the camera lost {lost} frames and the fusion expired a different set: {t}"
    );

    // The same statement, joined on sensor instants out of the evidence: a
    // count that matched by coincidence would fail here.
    let ev = evidence(&cwd, "slow");
    let instant = |r: &Vec<String>| -> i64 { ev.col(r, "tov_start_ns").parse().unwrap() };
    let frames: Vec<i64> = ev
        .rows
        .iter()
        .filter(|r| ev.col(r, "edge") == "cam0" && ev.col(r, "outcome") == "Delivered")
        .map(instant)
        .collect();
    let dropped: std::collections::BTreeSet<i64> = ev
        .rows
        .iter()
        .filter(|r| ev.col(r, "edge") == "cam0->camdet" && ev.col(r, "outcome") != "Delivered")
        .map(instant)
        .collect();
    let sets: Vec<(i64, i64, String, String)> = ev
        .rows
        .iter()
        .filter(|r| ev.col(r, "edge") == "track" && ev.col(r, "stage") == "track")
        .map(|r| {
            (
                instant(r),
                ev.col(r, "tov_end_ns").parse().unwrap(),
                ev.col(r, "outcome"),
                ev.col(r, "reason"),
            )
        })
        .collect();
    assert_eq!(dropped.len() as u64, lost, "drop rows against the queue");
    for f in &dropped {
        let owner: Vec<_> = sets.iter().filter(|s| s.0 <= *f && *f < s.1).collect();
        assert_eq!(owner.len(), 1, "dropped frame at {f} lies in {owner:?}");
        assert_eq!(
            (owner[0].2.as_str(), owner[0].3.as_str()),
            ("Missing", "pair_dropped"),
            "the sweep of dropped frame {f} was not expired as pair_dropped"
        );
    }
    for set in sets.iter().filter(|s| s.2 == "Missing") {
        let inside: Vec<&i64> = frames
            .iter()
            .filter(|f| set.0 <= **f && **f < set.1)
            .collect();
        assert_eq!(inside.len(), 1, "expired sweep {set:?} holds {inside:?}");
        assert!(
            dropped.contains(inside[0]),
            "sweep {set:?} expired, but its frame was not one the camera dropped"
        );
    }

    // The lidar half is untouched: lanes 0-20 of every fused sweep, bit for bit.
    let (a, b) = (
        fused_by_sweep(&run_dir(&cwd, "plain")),
        fused_by_sweep(&run_dir(&cwd, "slow")),
    );
    assert_eq!(
        b.len() as u64,
        u64_of(&t, "completed"),
        "file against summary"
    );
    let mut compared = 0;
    for (sweep, (_, _, _, rows)) in &b {
        let Some((_, _, _, rows_a)) = a.get(sweep) else {
            continue;
        };
        compared += 1;
        assert_eq!(rows.len(), rows_a.len(), "sweep {sweep}: track count");
        for (x, y) in rows.iter().zip(rows_a) {
            assert!(
                x[..21]
                    .iter()
                    .zip(&y[..21])
                    .all(|(p, q)| p.to_bits() == q.to_bits()),
                "sweep {sweep}: the lidar lanes differ, so the camera is not the only difference"
            );
        }
    }
    assert!(
        compared > 50,
        "only {compared} sweeps compared, so the lanes say little"
    );
    // Positive control: the plain run fused more, so the slowing did something
    // the comparison above could have seen.
    let plain = u64_of(
        &summary(&cwd, "plain")["lidar"]["reduce"]["detect"]["track"],
        "completed",
    );
    assert!(
        plain > u64_of(&t, "completed"),
        "the plain run completed {plain} sets, no more than the slowed one: {t}"
    );
    println!(
        "camera dropped {lost} of {} frames at cam0->camdet, no proc; the fusion expired exactly those {lost} sweeps as pair_dropped; {compared} fused sweeps with lidar lanes identical to the plain run's ({plain} completed there)",
        frames.len()
    );
}
