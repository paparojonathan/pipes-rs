//! The fourth stream, end to end: the right colour camera (`image_03`), its
//! own detector stage, and what the fusion does with two cameras.
//!
//! What CI can check without the model is the switch: `--right-camera on`
//! is a request for a second camera's detections, so a run that cannot run a
//! detector, or a drive without the camera, must stop before the clock
//! starts rather than fuse one camera and report two; and a run without the
//! detector replays no right camera under `auto`, and says so.
//!
//! What needs the model and drive_0005 is gated on the terms the other
//! real-drive tests are (`#[ignore]`, run by hand):
//!
//! ```text
//! cargo test --release -p pipes --test right_camera -- --ignored --nocapture
//! ```
//!
//! It runs the real binary over the whole drive twice: healthy, where both
//! cameras see every frame and the fusion confirms classes only where they
//! agree, gives camera-only detections a stereo range and checks the lidar's
//! distance against it; and with the left detector slowed so that it loses
//! frames, where the sweeps whose left frame was lost are answered from the
//! right frame instead of expiring.

use std::path::{Path, PathBuf};
use std::process::Command;

use arrow::array::RecordBatch;
use arrow::ipc::reader::StreamReader;
use pipes_kitti::camdet::MODEL_FILE;
use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{assert_ok, evidence, pipes, run_dir, run_json, stderr, stdout, summary, u64_of};

const N: usize = 8;
const PERIOD_NS: i64 = 5_000_000;

/// A lidar fixture with its calibration and the right camera.
fn two_camera_fx() -> FixtureDrive {
    let fx = FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    fx.write_right_camera().unwrap();
    fx
}

#[test]
fn the_right_camera_is_refused_where_nothing_reads_it_or_the_drive_lacks_it() {
    // With the camera on disk but no detector to read it: refused, naming
    // the detector.
    let fx = two_camera_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "nodet",
            "--track",
            "on",
            "--detector",
            "off",
            "--right-camera",
            "on",
        ],
    );
    assert!(!o.status.success(), "{}", stdout(&o));
    assert!(
        stderr(&o).contains("--right-camera on needs the detector"),
        "{}",
        stderr(&o)
    );
    assert!(!run_dir(&cwd, "nodet").exists());
    // With the detector asked for but no `image_03/`: refused, naming the
    // directory, before the model is even looked for (the test's CWD has
    // none).
    let fx = FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "nocam",
            "--track",
            "on",
            "--detector",
            "on",
            "--right-camera",
            "on",
        ],
    );
    assert!(!o.status.success(), "{}", stdout(&o));
    assert!(
        stderr(&o).contains("has no image_03/ directory"),
        "{}",
        stderr(&o)
    );
    assert!(!run_dir(&cwd, "nocam").exists());
}

#[test]
fn without_the_detector_the_right_camera_is_not_replayed_and_the_run_says_so() {
    let fx = two_camera_fx();
    let cwd = TempDir::new().unwrap();
    for (name, extra) in [("auto", vec![]), ("off", vec!["--right-camera", "off"])] {
        let mut args = vec!["--name", name, "--track", "on", "--cap", "16"];
        args.extend(extra);
        let o = pipes(&cwd, &fx, &args);
        assert_ok(&o);
        let out = stdout(&o);
        // Not a word about a stream the run did not have.
        assert!(
            !out.lines().any(|l| l.starts_with("cam1 ")
                || l.contains("driver=cam1")
                || l.contains("cam1->camdet")
                || l.contains("camdet-right")
                || l.contains("right camera:")),
            "{out}"
        );
        let csv = evidence(&cwd, name);
        assert!(csv.with("edge", "cam1").is_empty());
        assert!(csv.with("edge", "cam1_det->track").is_empty());
        let s = summary(&cwd, name);
        assert!(s["cam1"].is_null() && s["camera_detector_right"].is_null());
        let t = s["lidar"]["reduce"]["detect"]["track"].clone();
        assert_eq!(t["right_camera"], serde_json::json!(false));
        assert_eq!(u64_of(&t, "cam1_delivered"), 0);
        assert_eq!(u64_of(&t, "sets_from_right_camera"), 0);
        assert_eq!(t["fusion"]["right_camera"], serde_json::json!(false));
        assert!(t["fusion"]["stereo_rule"].is_null());
        assert!(t["fusion"]["stereo_check"].is_null());
        // And every set is the left camera's, with the right half absent.
        let r = run_json(&cwd, name);
        assert_eq!(r["right_camera"], serde_json::json!(name));
        assert!(r["n_cam1"].is_null() && r["stereo_rule"].is_null());
        for b in fused_batches(&run_dir(&cwd, name)) {
            assert_eq!(
                pipes_kitti::track::cam_side(&b),
                Some(pipes_kitti::cam0::Camera::Left)
            );
            assert_eq!(pipes_kitti::track::cam1_seq(&b), Some(-1));
            assert_eq!(
                pipes_kitti::track::cam1_pair_outcome(&b),
                Some("pair_absent")
            );
        }
    }
}

/// Every set of a run's `fused.arrows`, in order.
fn fused_batches(dir: &Path) -> Vec<RecordBatch> {
    let f = std::fs::File::open(dir.join("fused.arrows")).expect("fused.arrows");
    StreamReader::try_new(f, None)
        .expect("an Arrow IPC stream")
        .map(|b| b.expect("a batch"))
        .collect()
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
fn real_run(cwd: &Path, name: &str, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_pipes"))
        .current_dir(cwd)
        .args(["run", "--kitti-root"])
        .arg(kitti_root())
        .args(["--name", name, "--rerun", "null", "--dashboard", "off"])
        .args(extra)
        .output()
        .expect("spawn pipes")
}

/// **Two cameras, as a test.** The healthy run: both cameras see every frame
/// of drive 0005, every sweep pairs with both, the fusion keeps a class only
/// where the right camera confirmed the left's match, and every confirmed
/// track's lidar distance is checked against the two cameras' stereo range.
/// Then the slow left detector: with its one-frame queue dropping frames
/// (`--consumer-delay-ms 100 --cap 1`), the sweeps whose left frame was lost
/// are answered from the right camera's frame -- every match one camera's,
/// and the set saying so -- where one camera alone expired them.
#[test]
#[ignore = "needs drive_0005 and models/yolox_nano.onnx (scripts/fetch_model.ps1)"]
fn two_cameras_confirm_the_classes_and_the_right_one_answers_when_the_left_frame_is_lost() {
    use pipes_kitti::cam0::Camera;
    use pipes_kitti::track::{cam1_pair_outcome, cam1_seq, cam_seq, cam_side, track_sweep_seq};
    let cwd = TempDir::new().unwrap();
    let model = cwd.path().join(MODEL_FILE);
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::copy(repo_root().join(MODEL_FILE), &model)
        .unwrap_or_else(|e| panic!("{}: {e}", repo_root().join(MODEL_FILE).display()));

    // ---- healthy -------------------------------------------------------
    let o = real_run(cwd.path(), "both", &[]);
    // Exit 0 also says every INVARIANT line was OK, the right camera's
    // included: a FAIL exits 2.
    assert!(
        o.status.success(),
        "both: exit {:?}\n{}\n{}",
        o.status.code(),
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let out = String::from_utf8_lossy(&o.stdout).into_owned();
    // The right camera's own lines, OK. The driver's count is exact; the
    // stage's and the edges' are checked below with the slack a loaded host
    // needs, since the detector's one-frame queue can lose a frame to a
    // hiccup, declared, without the run being wrong.
    assert!(
        out.lines()
            .any(|l| l == "INVARIANT driver=cam1 admitted=154 missing=0 n_frames=154 -> OK"),
        "{out}"
    );
    for prefix in [
        "INVARIANT stage=camdet-right ",
        "INVARIANT edge=cam1->camdet ",
        "INVARIANT edge=cam1_det->track ",
        "INVARIANT stage=track right_camera ",
    ] {
        assert!(
            out.lines()
                .any(|l| l.starts_with(prefix) && l.ends_with("-> OK")),
            "no OK line starting {prefix:?} in:\n{out}"
        );
    }
    let s = summary(&cwd, "both");
    assert_eq!(u64_of(&s["cam1"], "admitted"), 154);
    assert!(
        u64_of(&s["camera_detector_right"], "produced") >= 150,
        "{s}"
    );
    let t = s["lidar"]["reduce"]["detect"]["track"].clone();
    assert_eq!(t["right_camera"], serde_json::json!(true));
    assert!(u64_of(&t, "cam1_delivered") >= 150, "{t}");
    assert!(u64_of(&t, "right_paired") > 140, "{t}");
    assert_eq!(u64_of(&t, "sets_from_right_camera"), 0);
    let f = t["fusion"].clone();
    assert_eq!(f["right_camera"], serde_json::json!(true));
    assert!(f["stereo_rule"].as_str().unwrap().contains("one_camera"));
    let c = f["completed"].clone();
    assert!(u64_of(&c, "right_samples") > 140, "{c}");
    assert!(u64_of(&c, "confirmed") > 0, "{c}");
    assert_eq!(u64_of(&c, "fused"), u64_of(&c, "confirmed"), "{c}");
    assert_eq!(
        u64_of(&c, "one_camera"),
        u64_of(&c, "unconfirmed") + u64_of(&c, "right_only"),
        "{c}"
    );
    // The agreement bites: some of the left's matches were not confirmed.
    assert!(u64_of(&c, "unconfirmed") > 0, "{c}");
    // Two sensors on one object: the stereo range lands on the lidar box's
    // centre, and reads farther than its near face by about the box's half
    // depth, as a silhouette's disparity should.
    let sc = f["stereo_check"].clone();
    assert!(u64_of(&sc, "tracks") > 50, "{sc}");
    let centre = sc["median_ratio_centre"].as_f64().unwrap();
    assert!(
        (0.9..1.1).contains(&centre),
        "median stereo/centre ratio {centre}: {sc}"
    );
    let near = sc["median_ratio"].as_f64().unwrap();
    assert!(
        (1.0..1.3).contains(&near),
        "median stereo/near-face ratio {near}: {sc}"
    );
    assert!(sc["within_fraction_centre"].as_f64().unwrap() > 0.4, "{sc}");
    // Every set is the left's, with the right frame of the same number.
    for b in fused_batches(&run_dir(&cwd, "both")) {
        assert_eq!(cam_side(&b), Some(Camera::Left));
        if cam1_pair_outcome(&b) == Some("paired") {
            assert_eq!(cam1_seq(&b), cam_seq(&b), "sweep {:?}", track_sweep_seq(&b));
        }
    }
    let r = run_json(&cwd, "both");
    assert_eq!(r["n_cam1"], serde_json::json!(154));
    assert!(r["stereo_rule"].as_str().is_some());
    println!(
        "both: confirmed {} of {} left matches, one_camera {}, stereo-ranged camera-only {} (dropped {}), stereo check {}",
        u64_of(&c, "confirmed"),
        u64_of(&c, "confirmed") + u64_of(&c, "unconfirmed"),
        u64_of(&c, "one_camera"),
        u64_of(&c, "stereo_matched"),
        u64_of(&c, "camera_only_dropped"),
        sc
    );

    // ---- the slow left detector -----------------------------------------
    let o = real_run(
        cwd.path(),
        "slow",
        &[
            "--consumer-delay-ms",
            "100",
            "--cap",
            "1",
            "--pair-wait-ms",
            "300",
        ],
    );
    assert!(
        o.status.success(),
        "slow: exit {:?}\n{}\n{}",
        o.status.code(),
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let s = summary(&cwd, "slow");
    let lost = common::dropped_on(&s, "cam0->camdet");
    assert!(lost > 0, "slowing the left detector cost it no frames: {s}");
    assert!(
        common::dropped_on(&s, "cam1->camdet") <= 3,
        "the right detector, which was not slowed, lost frames: {s}"
    );
    let t = s["lidar"]["reduce"]["detect"]["track"].clone();
    let from_right = u64_of(&t, "sets_from_right_camera");
    assert!(
        from_right > 0,
        "no sweep was answered from the right camera: {t}"
    );
    // The sweeps the left frame was lost for were answered, not expired:
    // what one camera alone could not do.
    assert!(
        from_right > u64_of(&t, "expired_dropped"),
        "more sweeps expired for a dropped left frame ({}) than were answered from the right ({from_right}): {t}",
        u64_of(&t, "expired_dropped")
    );
    assert_eq!(
        u64_of(&t, "completed")
            + u64_of(&t, "degraded")
            + u64_of(&t, "expired")
            + u64_of(&t, "errors"),
        u64_of(&t, "delivered")
    );
    // Those sets say so: built on the right frame, every match one camera's.
    let mut right_sets = 0u64;
    for b in fused_batches(&run_dir(&cwd, "slow")) {
        if cam_side(&b) == Some(Camera::Right) {
            right_sets += 1;
            assert_eq!(cam1_seq(&b), cam_seq(&b));
            assert_eq!(pipes_kitti::track::fused_count(&b), Some(0));
        }
    }
    assert_eq!(right_sets, from_right);
    let csv = evidence(&cwd, "slow");
    assert!(csv.with("edge", "cam1_det->track").len() >= 150);
    println!(
        "slow: left frames lost {lost}, sweeps answered from the right camera {from_right}, expired dropped {}",
        u64_of(&t, "expired_dropped")
    );

    // ---- a camera stream that breaks ------------------------------------
    // Either detector slowed to two seconds a frame: for every practical
    // purpose that stream is dead -- it delivers a frame every twenty
    // sweeps, each too old to pair. The fusion must declare it silent
    // after three sweeps without an arrival, stop waiting for it, and carry
    // on with the other camera at full rate with the classes trusted -- the
    // sets saying which camera was silent -- and the lidar chain behind it
    // must lose nothing. Each arrival is a sign of life: the next sweeps
    // wait for the camera again, and their matches are demoted until it
    // falls silent once more. So every sweep is one or the other.
    for (name, extra, broken) in [
        ("broken-left", ["--consumer-delay-ms", "2000"], Camera::Left),
        ("broken-right", ["--right-delay-ms", "2000"], Camera::Right),
    ] {
        let o = real_run(cwd.path(), name, &extra);
        assert!(
            o.status.success(),
            "{name}: exit {:?}\n{}\n{}",
            o.status.code(),
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        let s = summary(&cwd, name);
        let t = s["lidar"]["reduce"]["detect"]["track"].clone();
        assert_eq!(u64_of(&t, "delivered"), 154, "{name}: {t}");
        assert_eq!(u64_of(&t, "expired"), 0, "{name}: a sweep expired: {t}");
        assert_eq!(
            common::dropped_on(&s, "obj->track"),
            0,
            "{name}: the lidar chain lost sweeps waiting for the broken camera: {s}"
        );
        let silent_key = match broken {
            Camera::Left => "left_silent_sweeps",
            Camera::Right => "right_silent_sweeps",
        };
        let silent_sweeps = u64_of(&t, silent_key);
        assert!(
            silent_sweeps >= 100,
            "{name}: {silent_key} {silent_sweeps}: {t}"
        );
        let from_right = u64_of(&t, "sets_from_right_camera");
        match broken {
            Camera::Left => assert!(from_right >= 140, "{name}: {t}"),
            Camera::Right => assert_eq!(from_right, 0, "{name}: {t}"),
        }
        let c = t["fusion"]["completed"].clone();
        assert!(
            u64_of(&c, "fused") > 100,
            "{name}: the working camera's classes were not trusted: {c}"
        );
        // A sweep the broken camera was not silent on -- the three after
        // each of its rare arrivals -- waited for it, got nothing, and was
        // demoted; every other was silent. Nothing else.
        assert_eq!(
            silent_sweeps + u64_of(&c, "demoted_sets"),
            u64_of(&t, "delivered"),
            "{name}: {t}"
        );
        // Every set of a silent sweep says so, and is built on the other
        // camera's frame.
        let mut silent_sets = 0u64;
        for b in fused_batches(&run_dir(&cwd, name)) {
            let (left, right) = (
                pipes_kitti::track::cam0_pair_outcome(&b),
                cam1_pair_outcome(&b),
            );
            match broken {
                Camera::Left if left == Some("pair_silent") => {
                    silent_sets += 1;
                    assert_eq!(cam_side(&b), Some(Camera::Right));
                }
                Camera::Right if right == Some("pair_silent") => {
                    silent_sets += 1;
                    assert_eq!(cam_side(&b), Some(Camera::Left));
                }
                _ => {}
            }
        }
        assert_eq!(silent_sets, silent_sweeps, "{name}");
        println!(
            "{name}: silent on {silent_sweeps} sweeps, sets from the right {from_right}, fused {}, demoted sets {}, obj->track evicted 0",
            u64_of(&c, "fused"),
            u64_of(&c, "demoted_sets")
        );
    }
}
