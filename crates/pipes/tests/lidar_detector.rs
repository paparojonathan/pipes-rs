//! The `--lidar-detector` switch, end to end.
//!
//! What CI can check without the model is the refusal, on the terms
//! `tests/detector.rs` checks the camera detector's: `--lidar-detector on` is
//! a request for the lidar to say what it SAW, and a run that cannot load
//! the frozen model must stop -- naming how to get it -- rather than carry
//! on with unnamed tracks and report a detector that never ran as one that
//! found nothing. The switch is also scoped: its boxes have one consumer,
//! the fusion, and a run without the fusion is refused it.
//!
//! What needs the model and drive_0005 is gated, like the other real-drive
//! tests (`#[ignore]`, run by hand or by `docker compose run --rm test`):
//!
//! ```text
//! cargo test --release -p pipes --test lidar_detector -- --ignored --nocapture
//! ```
//!
//! Both gated tests run the network on the CPU (`--lidar-backend ort`),
//! which is what the test image has. That takes about 240 ms a sweep, so
//! two things are stretched to fit it: the replay runs at a quarter of real
//! time (a sweep every 414 ms), and the fusion waits up to 400 ms for a
//! sweep's boxes instead of one sweep span (`--pair-wait-ms 400`). At the
//! default wait every batch of a CPU run is `lidar_late`, at any replay
//! rate: the wait is host time, and the network does not get faster when
//! the drive is replayed slower. The GPU's timings are measured by hand and
//! are in docs/measurements.md.
//!
//! * [`the_lidar_detector_names_tracks_and_every_invariant_holds`] pins
//!   that the boxes reach the fusion, that tracks take their classes, and
//!   that every invariant holds.
//! * [`a_lidar_detector_that_stops_answering_does_not_stop_the_answer`]
//!   breaks the detector with `--lidar-delay-ms` and pins that the fusion
//!   says so sweep by sweep, names nothing, and still answers.
//! * [`the_second_view_finds_what_is_behind_and_leaves_what_is_ahead_alone`]
//!   runs the network both ways on three sweeps (`pipes bench-lidar --views
//!   around`): boxes behind the sensor appear, the ones well ahead are the
//!   one-view run's to the printed digit, and both runs repeat.

use std::path::{Path, PathBuf};
use std::process::Command;

use pipes_kitti::pillars::{MODEL_FILE, MODEL_FILE_HALF, MODEL_SHA256, MODEL_SHA256_HALF};
use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{assert_ok, line_with, pipes, run_dir, stderr, stdout, summary, u64_of};

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
fn a_missing_lidar_model_stops_the_run_and_names_how_to_get_it() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "nomodel",
            "--track",
            "on",
            "--lidar-detector",
            "on",
        ],
    );
    assert!(
        !o.status.success(),
        "a run with `--lidar-detector on` and no model SUCCEEDED:\n{}",
        stdout(&o)
    );
    let err = stderr(&o);
    assert!(err.contains("--lidar-detector on needs"), "{err}");
    // The GPU, the default, loads the file of 16-bit weights.
    assert!(
        err.contains(MODEL_FILE_HALF),
        "the error does not name the file:\n{err}"
    );
    // Both ways to the file: the image build, and the script by name.
    assert!(err.contains("docker compose build"), "{err}");
    assert!(
        err.contains("scripts/prepare_models.py pointpillars-kitti-fp16"),
        "{err}"
    );
    // The CPU loads the 32-bit file, and is told about that one.
    let on_cpu = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "nomodel",
            "--track",
            "on",
            "--lidar-detector",
            "on",
            "--lidar-backend",
            "ort",
        ],
    );
    assert!(!on_cpu.status.success(), "{}", stdout(&on_cpu));
    let err = stderr(&on_cpu);
    assert!(err.contains(MODEL_FILE), "{err}");
    assert!(
        err.contains("scripts/prepare_models.py pointpillars-kitti`"),
        "{err}"
    );
    // Refused before anything was written: no run directory at all, so no
    // half-run can later be read as a run.
    assert!(
        !run_dir(&cwd, "nomodel").exists(),
        "the refused run left a directory behind"
    );
    // Positive control: the same fixture and flags without the lidar
    // detector run, and say it was off.
    let o2 = pipes(&cwd, &fx, &["--name", "ref", "--track", "on"]);
    assert_ok(&o2);
    let s = summary(&cwd, "ref");
    assert!(s["lidar_detector"].is_null(), "{}", s["lidar_detector"]);
}

/// A file that is there but is not the pinned model is refused by its hash.
#[test]
fn a_lidar_model_that_is_not_the_pinned_one_is_refused() {
    // Each backend has its own file and its own pinned hash: 16-bit weights
    // on the GPU, the default, and 32-bit on the CPU.
    for (file, pinned, backend) in [
        (MODEL_FILE_HALF, MODEL_SHA256_HALF, "cuda"),
        (MODEL_FILE, MODEL_SHA256, "ort"),
    ] {
        let fx = fused_fx();
        let cwd = TempDir::new().unwrap();
        let path = cwd.path().join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not the model").unwrap();
        let o = pipes(
            &cwd,
            &fx,
            &[
                "--name",
                "wrong",
                "--track",
                "on",
                "--lidar-detector",
                "on",
                "--lidar-backend",
                backend,
            ],
        );
        assert!(!o.status.success(), "a wrong model file was accepted");
        let err = stderr(&o);
        assert!(err.contains("sha256"), "{err}");
        assert!(
            err.contains(pinned),
            "the error does not say what was expected:\n{err}"
        );
        assert!(!run_dir(&cwd, "wrong").exists());
    }
}

/// The detector's boxes have one consumer. Without the fusion the switch is
/// refused, before the model is looked for -- so the message is about the
/// switch, not about a file.
#[test]
fn the_lidar_detector_is_refused_where_nothing_reads_its_boxes() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "alone",
            "--track",
            "off",
            "--lidar-detector",
            "on",
        ],
    );
    assert!(!o.status.success(), "{}", stdout(&o));
    let err = stderr(&o);
    assert!(
        err.contains("--lidar-detector on needs the fusion (--track on"),
        "{err}"
    );
    assert!(!err.contains(MODEL_FILE), "{err}");
    assert!(!run_dir(&cwd, "alone").exists());
}

// ---- gated: the model and drive_0005 -------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn kitti_root() -> PathBuf {
    std::env::var("PIPES_KITTI_ROOT")
        .map_or_else(|_| repo_root().join("../data/kitti"), PathBuf::from)
}

/// A temp CWD holding a copy of the lidar model, where the binary looks.
fn cwd_with_model() -> TempDir {
    let cwd = TempDir::new().unwrap();
    let model = cwd.path().join(MODEL_FILE);
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::copy(repo_root().join(MODEL_FILE), &model)
        .unwrap_or_else(|e| panic!("{}: {e}", repo_root().join(MODEL_FILE).display()));
    cwd
}

/// One real run of drive_0005 in `cwd` with the lidar detector on the CPU,
/// the cameras' detector off (it has its own tests, and its own model), at a
/// quarter of real time, the fusion waiting up to 400 ms for the boxes.
fn real_run(cwd: &Path, name: &str, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_pipes"))
        .current_dir(cwd)
        .args(["run", "--kitti-root"])
        .arg(kitti_root())
        .args(["--name", name, "--rerun", "null", "--dashboard", "off"])
        .args([
            "--detector",
            "off",
            "--rate",
            "0.25",
            "--pair-wait-ms",
            "400",
        ])
        .args(["--lidar-detector", "on", "--lidar-backend", "ort"])
        .args(extra)
        .output()
        .expect("spawn pipes")
}

/// Every `INVARIANT` line of a run's output ends `-> OK`; returns how many.
fn all_invariants_hold(out: &str) -> usize {
    let lines: Vec<&str> = out.lines().filter(|l| l.starts_with("INVARIANT")).collect();
    for l in &lines {
        assert!(l.ends_with("-> OK"), "{l}");
    }
    lines.len()
}

#[test]
#[ignore = "needs drive_0005, models/pointpillars-kitti.onnx and ONNX Runtime at ORT_DYLIB_PATH"]
fn the_lidar_detector_names_tracks_and_every_invariant_holds() {
    let cwd = cwd_with_model();
    let o = real_run(cwd.path(), "named", &[]);
    assert!(
        o.status.success(),
        "exit {:?}\n{}\n{}",
        o.status.code(),
        stdout(&o),
        stderr(&o)
    );
    let out = stdout(&o);
    assert!(all_invariants_hold(&out) > 20, "{out}");
    for line in [
        "INVARIANT stage=pillars ",
        "INVARIANT edge=velo->pillars ",
        "INVARIANT edge=pillar_det->track ",
        "INVARIANT stage=track lidar_detector ",
    ] {
        line_with(&out, line);
    }
    assert!(
        out.contains("pillars read the velodyne driver's buffer: OK"),
        "the detector did not read the sweeps in place:\n{out}"
    );

    let s = summary(&cwd, "named");
    let l = s["lidar_detector"].clone();
    assert_eq!(l["model_sha256"], serde_json::json!(MODEL_SHA256), "{l}");
    assert_eq!(l["backend"], serde_json::json!("ort"), "{l}");
    // On the CPU it looks ahead only: one pass is already longer than a sweep.
    assert_eq!(l["views"], serde_json::json!(["ahead"]), "{l}");
    let produced = u64_of(&l, "produced");
    assert!(produced > 100, "the detector ran on {produced} sweeps: {l}");

    // The names reached the tracks, and the far end counted the same ones.
    let n = s["lidar"]["reduce"]["detect"]["track"]["naming"].clone();
    assert!(u64_of(&n, "named_sweeps") > 100, "{n}");
    let named = u64_of(&n["completed"], "named") + u64_of(&n["degraded"], "named");
    assert!(named > 500, "{named} tracks named: {n}");
    assert_eq!(u64_of(&n, "down_named"), named, "{n}");
    for claim in ["wrong_seq", "unknown", "stray"] {
        assert_eq!(u64_of(&n["join"], claim), 0, "{claim}: {n}");
    }

    // And the answers were made on those tracks.
    let st = s["lidar"]["reduce"]["detect"]["track"]["state"].clone();
    let answers = u64_of(&st, "down_delivered");
    assert!(answers > 100, "{st}");
    println!("{produced} sweeps detected, {named} tracks named, {answers} answers");
}

#[test]
#[ignore = "needs drive_0005, models/pointpillars-kitti.onnx and ONNX Runtime at ORT_DYLIB_PATH"]
fn a_lidar_detector_that_stops_answering_does_not_stop_the_answer() {
    let cwd = cwd_with_model();
    // Two seconds a sweep: every batch arrives after the fusion has given
    // up on it, or never leaves the stage's queue.
    let o = real_run(cwd.path(), "broken", &["--lidar-delay-ms", "2000"]);
    assert!(
        o.status.success(),
        "exit {:?}\n{}\n{}",
        o.status.code(),
        stdout(&o),
        stderr(&o)
    );
    let out = stdout(&o);
    assert!(all_invariants_hold(&out) > 20, "{out}");
    let s = summary(&cwd, "broken");
    let t = s["lidar"]["reduce"]["detect"]["track"].clone();
    let n = t["naming"].clone();
    let delivered = u64_of(&t, "delivered");
    assert!(delivered > 100, "{t}");
    // Every sweep got a verdict on its batch, and almost none of them was
    // "named": late, dropped, or the detector declared silent.
    let (named, late, dropped, silent) = (
        u64_of(&n, "named_sweeps"),
        u64_of(&n, "late_sweeps"),
        u64_of(&n, "dropped_sweeps"),
        u64_of(&n, "silent"),
    );
    assert!(
        named <= 2 && late + dropped + silent >= delivered - 2,
        "named {named} late {late} dropped {dropped} silent {silent} of {delivered}: {n}"
    );
    assert!(silent > 0, "the fusion never stopped waiting: {n}");
    // And the chain carried on: an answer for the sweeps the fusion
    // completed, without the lidar's names.
    let st = t["state"].clone();
    let answers = u64_of(&st, "down_delivered");
    assert!(answers > 100, "{st}");
    println!("{delivered} sweeps: named {named}, late {late}, dropped {dropped}, silent {silent}");
}

/// `pipes bench-lidar` over drive_0005's first three sweeps on the CPU,
/// looking `views`: its output, and the lines of sweep 2's boxes.
fn bench_boxes(cwd: &Path, views: &str) -> (String, Vec<String>) {
    let o = Command::new(env!("CARGO_BIN_EXE_pipes"))
        .current_dir(cwd)
        .args(["bench-lidar", "--kitti-root"])
        .arg(kitti_root())
        .args(["--backend", "ort", "--sweeps", "3", "--show", "2"])
        .args(["--views", views])
        .output()
        .expect("spawn pipes");
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    let boxes = out
        .lines()
        .filter(|l| l.contains("centre ("))
        .map(str::to_string)
        .collect();
    (out, boxes)
}

/// The x of a printed box's centre, metres ahead of the sensor.
fn centre_x(line: &str) -> f32 {
    let rest = line.split("centre (").nth(1).unwrap();
    rest.split(',').next().unwrap().trim().parse().unwrap()
}

#[test]
#[ignore = "needs drive_0005, models/pointpillars-kitti.onnx and ONNX Runtime at ORT_DYLIB_PATH"]
fn the_second_view_finds_what_is_behind_and_leaves_what_is_ahead_alone() {
    let cwd = cwd_with_model();
    let (out_ahead, ahead) = bench_boxes(cwd.path(), "ahead");
    let (out_around, around) = bench_boxes(cwd.path(), "around");
    assert!(out_ahead.contains("looking ahead;"), "{out_ahead}");
    assert!(
        out_around.contains("looking ahead, behind;"),
        "{out_around}"
    );
    for out in [&out_ahead, &out_around] {
        assert!(out.contains("repeatable: yes"), "{out}");
    }
    // One view finds nothing behind the sensor; two find things both ways.
    assert!(!ahead.is_empty(), "{out_ahead}");
    assert!(ahead.iter().all(|l| centre_x(l) > -5.0), "{out_ahead}");
    let behind = around.iter().filter(|l| centre_x(l) < -5.0).count();
    assert!(behind > 0, "nothing found behind the sensor: {out_around}");
    // And what is well ahead is found exactly as it was: the second view
    // adds boxes, it does not move the first view's.
    let well_ahead: Vec<&String> = ahead.iter().filter(|l| centre_x(l) > 8.0).collect();
    assert!(!well_ahead.is_empty(), "{out_ahead}");
    for l in &well_ahead {
        assert!(around.contains(l), "missing from the two-view run: {l}");
    }
    println!(
        "sweep 2: {} boxes looking ahead, {} looking both ways, {behind} of them behind",
        ahead.len(),
        around.len()
    );
}
