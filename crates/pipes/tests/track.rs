//! The last two links, end to end, with no dataset: the **fusion** and the
//! **answer**.
//!
//! Every earlier link in this chain consumed one stream. `track` consumes two
//! — a lidar-derived one it blocks on and a camera-derived one it drains — and
//! that is where the new ways to be wrong are. Six of them, and a test for
//! each:
//!
//! * **A fusion that never actually looks at the camera** would pair, produce
//!   and balance exactly as this one does. [`the_camera_half_is_joined_back`]
//!   pins that every fused sample names a frame the camera driver really
//!   admitted **and whose instant falls inside that sweep**, which is the
//!   pairing rule itself. This is the test that exists because
//!   `Sample::parent` was once declared, written nowhere and read nowhere
//!   while a derived sample naming the wrong sweep passed the whole suite.
//! * **A fusion that refuses nothing** cannot be told from one that never had
//!   to. [`a_missing_camera_frame_expires_the_set_rather_than_faking_it`]
//!   removes the camera's contribution and pins that the sets EXPIRE, with a
//!   reason, and that the answers vanish with them.
//! * **A stage that copied its input** would produce every row and every
//!   invariant. [`the_fusion_borrows_both_inputs`] measures both halves.
//! * **A camera edge whose rows go missing** breaks nothing visible until an
//!   invariant is read. [`every_camera_reference_owes_a_row`] pins that one,
//!   including the references that were never paired.
//! * **A switch that quietly changed something else** is the failure this
//!   project has the most history with. [`track_off_changes_nothing`] pins
//!   that the chain under `--track off` is what it was, and
//!   [`the_camera_path_is_untouched_by_the_fusion`] pins the one number the
//!   camera path is measured by, `proc bytes/frame`.
//! * **A calibration that silently defaulted** produces an overlay that looks
//!   right and is 6.58 px wrong.
//!   [`a_missing_calibration_fails_the_run_rather_than_defaulting`] pins the
//!   refusal.
//!
//! **Nothing here asserts a track count or a distance.** Both are properties
//! of the scene, and this fixture's scene is a line of points — chosen so the
//! arithmetic is checkable, not so the output looks impressive. The measured
//! numbers on the real drive are produced by `pipes-kitti`'s own ignored
//! `track_real_drive` harness, which also checks the answer against the raw
//! cloud.

use pipes_core::sample::StreamId;
use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{
    assert_line, assert_ok, dropped_on, evidence, pipes, proc_stage, stdout, summary, u64_of,
};

/// Frames and sweeps per fixture run, and the period they share.
const N: usize = 8;
const PERIOD_NS: i64 = 5_000_000;

/// The period of the stale-window test, which replays PACED so that the
/// fusion's bounded wait is real: 25 ms, a 200 ms run, the same period
/// `lidar.rs` gives its paced test and for the same reason. At 5 ms a loaded
/// host once stalled the velodyne driver's thread for about 35 ms after sweep
/// 1, every later sweep was `deadline_skipped`, and the fusion saw two sweeps
/// -- both with frames -- so the six the fixture starves never reached it. At
/// 25 ms the same stall would have to reach 150 ms.
const PACED_PERIOD_NS: i64 = 25_000_000;

/// A lidar fixture **with** its date-level calibration, which `--track on`
/// requires.
fn fused_fx() -> FixtureDrive {
    let fx = FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    fx
}

/// The `lidar.reduce.detect.track` block of a run's `summary.json`.
fn track_block(cwd: &TempDir, name: &str) -> serde_json::Value {
    let s = summary(cwd, name);
    let t = s["lidar"]["reduce"]["detect"]["track"].clone();
    assert!(
        !t.is_null(),
        "no lidar.reduce.detect.track block in summary.json — the fusion did not run:\n{s}"
    );
    t
}

/// The fusion's provenance join, which is the whole reason the camera half is
/// allowed to travel in the payload rather than in `Sample::parent`.
///
/// Two claims, and the second is the one a fake would fail: the named frame
/// was really admitted by the camera driver, **and** its instant is inside
/// that sweep's range. A fusion that stamped a plausible number would pass the
/// first.
#[test]
fn the_camera_half_is_joined_back() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "join", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert!(
        out.lines()
            .any(|l| l.starts_with("fused cam_seq = a frame the camera really admitted")),
        "the camera half of the provenance was not joined:\n{out}"
    );
    let t = track_block(&cwd, "join");
    assert_eq!(
        t["cam_pair_join_ok"],
        serde_json::json!(true),
        "the pair join did not pass"
    );
    // Positive control: a join over nothing also "passes" if nobody checks the
    // population, so pin that there was one.
    assert!(
        u64_of(&t, "completed") > 0,
        "nothing was paired, so the join above proved nothing: {t}"
    );
    assert_eq!(u64_of(&t, "degraded"), 0, "nothing asked for a stale pair");
    // The fixture's camera instant is the sweep's START, which a HALF-OPEN
    // containment test accepts and a strict one rejects -- and which a test
    // closed at the wrong end would hand to the PREVIOUS sweep. The positive
    // control above already says the first: on this fixture a strict test
    // completes nothing at all. The pair age says the second: every completed
    // pair is the frame at the start, half a period BEFORE the trigger, and
    // not the one at the end, half a period after it.
    let half_period_ms = PERIOD_NS as f64 / 2.0 / 1e6;
    for key in ["pair_age_ms_min", "pair_age_ms_max"] {
        assert_eq!(
            t[key].as_f64(),
            Some(-half_period_ms),
            "{key}: a completed pair is not the frame at the sweep's start: {t}"
        );
    }
    // A set that did expire did so because its frame had not ARRIVED within
    // the wait -- one sweep span, 5 ms here, which a loaded host can miss --
    // and never because the frame was produced and lost or the stream was
    // absent. This once asserted `expired == 0` as the half-open proof, and
    // read a late frame on a busy host as a refusal.
    let expired = u64_of(&t, "expired");
    assert_eq!(
        u64_of(&t, "expired_late"),
        expired,
        "a set expired for a reason other than a late frame: {t}"
    );
    assert_eq!(
        u64_of(&t, "completed") + expired + u64_of(&t, "errors"),
        u64_of(&t, "delivered"),
        "a sweep was neither paired nor expired: {t}"
    );
}

/// A camera frame the fusion cannot have must EXPIRE the set, not be invented.
///
/// `--pair-wait-ms 0` with `--consumer-delay-ms` past the point where `proc`
/// can keep up is the reachable version of "the frame is not there": the
/// camera path still delivers, the reference is simply not in hand when the
/// sweep is. The set must then produce nothing, say why, and take the answer
/// with it.
#[test]
fn a_missing_camera_frame_expires_the_set_rather_than_faking_it() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "expire",
            "--track",
            "on",
            "--cap",
            "16",
            "--rate",
            "1.0",
            "--consumer-delay-ms",
            "60",
            "--pair-wait-ms",
            "0",
        ],
    );
    assert_ok(&o);
    let t = track_block(&cwd, "expire");
    let expired = u64_of(&t, "expired");
    assert!(
        expired > 0,
        "nothing expired, so this test proves nothing about refusal: {t}"
    );
    // Refused, not faked: produced + expired accounts for every batch.
    assert_eq!(
        u64_of(&t, "produced") + expired + u64_of(&t, "errors"),
        u64_of(&t, "delivered"),
        "a sweep neither produced nor expired: {t}"
    );
    // And every sweep of the drive is accounted for, including any the
    // fusion was never handed: the sets it made plus the sweeps lost in
    // front of it are the drive, and the losses are named edge by edge.
    let b = &t["sweeps_before_track"];
    assert_eq!(u64_of(b, "sweeps"), N as u64, "{b}");
    assert_eq!(u64_of(b, "reached"), u64_of(&t, "delivered"), "{b}");
    assert_eq!(
        u64_of(&t, "completed")
            + u64_of(&t, "degraded")
            + expired
            + u64_of(&t, "errors")
            + u64_of(b, "skipped")
            + u64_of(b, "evicted_velo_reduce")
            + u64_of(b, "evicted_det_detect")
            + u64_of(b, "evicted_obj_track")
            + u64_of(b, "stage_errors"),
        N as u64,
        "a sweep is in no category: {t}"
    );
    assert!(
        stdout(&o).contains(&format!(
            "NEVER REACHED track {} of {N} sweeps",
            N as u64 - u64_of(&t, "delivered")
        )),
        "the fusion sets line does not account for every sweep:\n{}",
        stdout(&o)
    );
    // The reason is recorded and is one of the named ones, never blank.
    let ev = evidence(&cwd, "expire");
    let missing: Vec<_> = ev
        .rows
        .iter()
        .filter(|r| ev.col(r, "edge") == "track" && ev.col(r, "outcome") == "Missing")
        .collect();
    assert_eq!(missing.len() as u64, expired, "expired sets without a row");
    for r in &missing {
        let reason = ev.col(r, "reason");
        assert!(
            ["pair_late", "pair_dropped", "pair_absent"].contains(&reason.as_str()),
            "expired with reason {reason:?}, which attributes nothing"
        );
        // A set that was never produced has no payload and no arrival order.
        assert_eq!(ev.col(r, "payload_bytes"), "0");
        assert_eq!(ev.col(r, "arrival_seq"), "");
    }
    // And the answer went with it, rather than being produced from nothing:
    // `state` saw what `track` produced, less what `track->state` -- a cap-4
    // DropOldest edge -- evicted on the way, and not one sample more. (On a
    // run where every set expired nothing travels that edge at all, and it
    // has no `stages` entry; `dropped_on` reads that as 0.)
    assert_eq!(
        u64_of(&t["state"], "delivered") + dropped_on(&summary(&cwd, "expire"), "track->state"),
        u64_of(&t, "produced"),
        "the answer stage saw samples the fusion never produced, or lost some nobody counted: {t}"
    );
    // Positive control on the same fixture and the same flags but with the
    // wait the stage defaults to: the sets complete. Without this, "expired"
    // could be the only thing this pipeline ever does.
    // A wait long enough to cover the delay this fixture's `proc` is under.
    // The stage's own default is ONE SWEEP SPAN, and this fixture's sweep is
    // 5 ms against a 60 ms delay -- so the default cannot recover here and
    // asserting that it does would be asserting the fixture's timing rather
    // than the mechanism. The mechanism is: a bounded wait converts `late`
    // into `completed`, and its bound is declared.
    let o2 = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "waited",
            "--track",
            "on",
            "--cap",
            "16",
            "--rate",
            "1.0",
            "--consumer-delay-ms",
            "60",
            "--pair-wait-ms",
            "800",
        ],
    );
    assert_ok(&o2);
    let t2 = track_block(&cwd, "waited");
    assert!(
        u64_of(&t2, "completed") > u64_of(&t, "completed"),
        "the bounded wait recovered nothing: without {t}\nwith {t2}"
    );
}

/// The fusion must borrow both inputs and allocate only its own output.
///
/// Both halves are asserted because they are different claims and a stage that
/// failed either would still produce every row: 0 for the inputs says nothing
/// was copied, non-zero for the output says the stage genuinely transformed.
#[test]
fn the_fusion_borrows_both_inputs() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "borrow", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let t = track_block(&cwd, "borrow");
    assert_eq!(
        u64_of(&t, "read_bytes_per_sample"),
        0,
        "the fusion allocated while reading its inputs: {t}"
    );
    assert_eq!(
        u64_of(&t["state"], "read_bytes_per_sample"),
        0,
        "the answer stage allocated while reading the tracks: {t}"
    );
    assert_eq!(
        u64_of(&t["state"], "down_bytes_per_sample"),
        0,
        "the far end of the chain allocated per answer: {t}"
    );
    // The other half: the stages that build something must pay for it.
    assert!(
        u64_of(&t, "build_bytes_per_sample") > 0,
        "the fusion allocated NOTHING to build its output, which cannot be true: {t}"
    );
    assert!(
        u64_of(&t["state"], "build_bytes_per_sample") > 0,
        "the answer cost nothing to build, which cannot be true: {t}"
    );
    // And the buffers really were shared, at all three new links.
    for key in [
        "input_storage_id_equal",
        "output_storage_id_equal",
        "state_storage_id_equal",
        "parent_seq_equal",
        "state_parent_seq_equal",
    ] {
        assert_eq!(t[key], serde_json::json!(true), "{key} failed: {t}");
    }
}

/// `INVARIANT rows edge=cam_det->track` needs one row per admitted reference —
/// including every reference the fusion looked at and never paired.
#[test]
fn every_camera_reference_owes_a_row() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "rows", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    // The invariant itself, by name: it is the one that makes row counts mean
    // anything, and a carve-out for this edge would be invisible otherwise.
    assert_line(
        &o,
        &format!("INVARIANT rows edge=cam_det->track rows={N} admitted={N} -> OK"),
    );
    let ev = evidence(&cwd, "rows");
    let cam: Vec<_> = ev
        .rows
        .iter()
        .filter(|r| ev.col(r, "edge") == "cam_det->track")
        .collect();
    assert_eq!(cam.len(), N, "one row per camera reference");
    for r in &cam {
        assert_eq!(
            ev.col(r, "outcome"),
            "Delivered",
            "a reference that reached the stage was not Delivered"
        );
        assert_eq!(
            ev.col(r, "stream"),
            StreamId::CAM_DET.0.to_string(),
            "the camera reference travelled on the wrong stream"
        );
    }
    assert!(out.contains("cam_det->track"), "{out}");
}

/// Under `--track off` the run is what it was before this work: no fusion
/// edges, no fusion block, and the chain still ends at the detections.
///
/// Only that. This test once went on to run the same fixture with `--track
/// on` and require `detect`'s counts to match the `off` run key for key,
/// which pinned `off` as the default (it no longer is) and asserted the
/// scheduler: `velo->reduce` is a cap-2 DropOldest edge and the runs are
/// unpaced, so the two need not see the same sweeps, and `delivered 7
/// against 8` failed it one run in six. What `--track` must not change is
/// measured where a per-frame figure can carry the claim:
/// [`the_camera_path_is_untouched_by_the_fusion`].
#[test]
fn track_off_changes_nothing() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "off", "--cap", "16"]);
    assert_ok(&o);
    let out = stdout(&o);
    for edge in [
        "obj->track",
        "cam_det->track",
        "track->state",
        "state->sink",
    ] {
        assert!(
            !out.contains(edge),
            "`--track off` still opened {edge}:\n{out}"
        );
    }
    let s = summary(&cwd, "off");
    assert!(
        s["lidar"]["reduce"]["detect"]["track"].is_null(),
        "`--track off` still published a track block"
    );
    // And the chain did run up to the detections: the switch removed the
    // fusion, not the stage before it.
    assert!(
        !s["lidar"]["reduce"]["detect"].is_null(),
        "`--track off` took the detect block with it"
    );
}

/// The fusion takes its camera input from `proc`. This pins that taking it
/// changed nothing about what `proc` is measured as.
#[test]
fn the_camera_path_is_untouched_by_the_fusion() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let want = u64::from(pipes_kitti::testing::FIXTURE_W * pipes_kitti::testing::FIXTURE_H);
    for (name, args) in [
        ("plain", vec!["--name", "plain", "--cap", "16"]),
        (
            "fused",
            vec!["--name", "fused", "--track", "on", "--cap", "16"],
        ),
    ] {
        let o = pipes(&cwd, &fx, &args);
        assert_ok(&o);
        let s = summary(&cwd, name);
        let st = proc_stage(&s);
        assert_eq!(
            u64_of(&st, "delivered"),
            N as u64,
            "{name}: proc did not see every frame"
        );
        assert_eq!(
            st["bytes_alloc_per_frame_mean"].as_f64().unwrap_or(-1.0),
            want as f64,
            "{name}: proc's measured per-frame allocation moved"
        );
    }
    // The emission itself is real and is reported on its own, so "it did not
    // move" is not the same claim as "nothing happened".
    let t = track_block(&cwd, "fused");
    assert_eq!(
        u64_of(&t, "cam_delivered"),
        N as u64,
        "the fusion saw no camera references at all, so the test above is vacuous: {t}"
    );
    assert_eq!(u64_of(&t, "cam_bad_format"), 0);
}

/// `--track on` without a calibration must FAIL, not fall back.
///
/// The house rule that makes this worth a test: an identity for a missing
/// `R_rect_00` costs 6.58 px and a zero fourth column costs 2.91 px, and both
/// produce an overlay that looks correct. A silent default is the dangerous
/// outcome here, not a panic.
#[test]
fn a_missing_calibration_fails_the_run_rather_than_defaulting() {
    // Deliberately NOT `fused_fx()`: no calibration is written.
    let fx = FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "nocal", "--track", "on", "--cap", "16"],
    );
    assert!(
        !o.status.success(),
        "a run with `--track on` and no calibration SUCCEEDED:\n{}",
        stdout(&o)
    );
    let err = common::stderr(&o);
    assert!(
        err.contains("calibration") || err.contains("calib"),
        "the failure did not name the calibration:\n{err}"
    );
    // Positive control: the same fixture WITH the calibration runs.
    fx.write_calib().unwrap();
    let o2 = pipes(
        &cwd,
        &fx,
        &["--name", "cal", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o2);
}

/// **`--pair-stale-ms` had no test at all**, which is exactly the shape the
/// house rules forbid: `the_camera_half_is_joined_back` asserts
/// `degraded == 0`, and a stage that could never produce a degraded set at all
/// would pass it. This is the positive control for that assertion — the flag
/// is asked for by name, a degraded set is produced, and the `stale` label is
/// then chased to all three places a reader could look for it.
///
/// The scenario is **deterministic and needs no timing at all**, which matters
/// because every other way of starving the camera here is a race. The fixture
/// is written with two camera frames against eight sweeps: sweeps 2..7 have no
/// camera instant inside them because those frames were never recorded, not
/// because anything was late or dropped. The freshest frame is then older than
/// the sweep by a known amount and either the run declares a window that covers
/// it or the set expires. The pairing needs no timing; the paced REPLAY does,
/// which is why the fixture is written at [`PACED_PERIOD_NS`].
#[test]
fn a_declared_stale_window_produces_a_degraded_set_and_labels_it_everywhere() {
    // Two frames, eight sweeps, the same period: sweeps 0 and 1 contain a
    // camera instant and sweeps 2..7 cannot.
    const FRAMES: usize = 2;
    let fx = FixtureDrive::with_velodyne(FRAMES, N, PACED_PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    let cwd = TempDir::new().unwrap();
    // A wait far longer than the run, so a sweep whose frame is merely late is
    // never confused with one whose frame does not exist. It costs nothing:
    // the wait ends when the camera edge closes, and this camera has two
    // frames to produce.
    let args = |name: &'static str, stale: &'static str| -> Vec<&'static str> {
        vec![
            "--name",
            name,
            "--track",
            "on",
            "--cap",
            "16",
            "--rate",
            "1.0",
            "--pair-wait-ms",
            "2000",
            "--pair-stale-ms",
            stale,
        ]
    };

    // ---- without the window: the same sweeps EXPIRE ----------------------
    let o0 = pipes(&cwd, &fx, &args("norefuse", "0"));
    assert_ok(&o0);
    let t0 = track_block(&cwd, "norefuse");
    assert_eq!(
        u64_of(&t0, "degraded"),
        0,
        "a run that declared no stale window produced a degraded set: {t0}"
    );
    let expired0 = u64_of(&t0, "expired");
    assert!(
        expired0 > 0,
        "the fixture did not starve the fusion at all, so the run below proves nothing: {t0}"
    );

    // ---- with the window: the same sweeps DEGRADE -------------------------
    let o = pipes(&cwd, &fx, &args("stale", "1000"));
    assert_ok(&o);
    let t = track_block(&cwd, "stale");
    let degraded = u64_of(&t, "degraded");
    assert!(
        degraded > 0,
        "`--pair-stale-ms` produced no degraded set, so nothing here is tested: {t}"
    );
    // The accounting is asserted WITHIN each run and never across the two.
    // `obj->track` is a bounded queue with its own eviction policy, so the two
    // runs need not see the same number of sweeps; an equality between them
    // would be asserting the scheduler rather than the mechanism, and it
    // failed exactly that way before this comment existed.
    assert_eq!(
        u64_of(&t, "produced") + u64_of(&t, "expired") + u64_of(&t, "errors"),
        u64_of(&t, "delivered"),
        "a sweep neither produced nor expired: {t}"
    );
    assert_eq!(
        u64_of(&t, "completed") + degraded,
        u64_of(&t, "produced"),
        "a produced set was neither completed nor degraded: {t}"
    );
    // The window CONVERTED refusals rather than inventing sets: a window wider
    // than the whole fixture leaves nothing to expire, where the identical run
    // without one expired every sweep whose frame was never recorded.
    assert_eq!(
        u64_of(&t, "expired"),
        0,
        "a set expired although a stale window wider than the whole run was declared: {t}"
    );

    // ---- the label, in all three places it has to survive -----------------
    // 1. the far end of the chain, read back out of the ANSWER's payload.
    //    This is the one that proves `pair_outcome` survived `state` rebuilding
    //    the batch, rather than being a string `track` wrote and nobody kept.
    let st = t["state"].clone();
    let down_stale = u64_of(&st, "down_stale");
    assert!(
        down_stale > 0,
        "no answer at the far end knows it came from a stale frame: {st}"
    );
    assert!(
        down_stale <= degraded,
        "more answers claim a stale pair ({down_stale}) than the fusion produced degraded sets ({degraded}): {st}"
    );
    assert_eq!(
        u64_of(&st, "down_paired") + down_stale,
        u64_of(&st, "down_delivered"),
        "an answer was neither paired nor stale: {st}"
    );
    // Its negative control, on the run that declared no window: the column has
    // to be able to read 0, or `down_stale > 0` above is compatible with a
    // counter that is really counting answers.
    let st0 = t0["state"].clone();
    assert!(
        u64_of(&st0, "down_delivered") > 0,
        "the run without a window produced no answers, so the control below is vacuous: {st0}"
    );
    assert_eq!(
        u64_of(&st0, "down_stale"),
        0,
        "a run that declared no stale window still reported stale answers: {st0}"
    );

    // 2. `evidence.csv` alone. A degraded set is a Delivered row like every
    //    other one, and before this the staleness lived only in the payload:
    //    a reader with the evidence in front of them could not tell which
    //    answers came from a frame that was already old.
    let ev = evidence(&cwd, "stale");
    let stale_rows = ev
        .rows
        .iter()
        .filter(|r| {
            ev.col(r, "edge") == "track"
                && ev.col(r, "outcome") == "Delivered"
                && ev.col(r, "reason") == "stale"
        })
        .count() as u64;
    assert_eq!(
        stale_rows, degraded,
        "evidence.csv cannot tell a degraded set from a clean one"
    );
    // The positive control on the same column, taken from the run with no
    // window so it cannot depend on which sweeps survived: a clean pair leaves
    // `reason` BLANK, so a non-empty one means something rather than being
    // written on every row.
    let ev0 = evidence(&cwd, "norefuse");
    let clean: Vec<_> = ev0
        .rows
        .iter()
        .filter(|r| ev0.col(r, "edge") == "track" && ev0.col(r, "outcome") == "Delivered")
        .collect();
    assert!(
        !clean.is_empty(),
        "the run without a window produced no sets at all, so the control is vacuous"
    );
    for r in &clean {
        assert_eq!(
            ev0.col(r, "reason"),
            "",
            "a clean pair carries a reason, so `stale` distinguishes nothing"
        );
    }

    // 3. the run's own words, so an operator sees it without reading a file.
    let out = stdout(&o);
    assert!(
        out.contains("degraded"),
        "the run never mentions the degraded sets it produced:
{out}"
    );
    // ... and the answer's own words say it. The last sweep is a degraded
    // set's -- frame 1 is the newest the camera ever took -- and its answer
    // is printed leading with how stale; the run without a window ended on a
    // clean pair, whose words carry no such mark. Before this, a stale answer
    // printed exactly like a fresh one and only the frame number differed.
    let last = |out: &str| -> String {
        out.lines()
            .find_map(|l| l.strip_prefix("ANSWER last: "))
            .unwrap_or_else(|| panic!("no ANSWER last line:\n{out}"))
            .to_string()
    };
    let words = last(&out);
    assert!(
        words.starts_with("STALE camera ") && words.contains(" ms (frame 1): "),
        "the last answer came from a stale frame and does not say so: {words}"
    );
    let words0 = last(&stdout(&o0));
    assert!(
        !words0.contains("STALE"),
        "an answer from a clean pair says it was stale: {words0}"
    );
    // ... and how stale, in the summary. `pair_age_ms_*` covers completed
    // sets only, so a run of stale answers used to say `null` there and
    // nowhere else how old its camera was. Each stale sweep here took frame
    // 1, whose instant is sweep 1's start; sweep j's trigger is half a period
    // after sweep j's start, so its pair age is -((j - 1) * 25 + 12.5) ms:
    // -37.5 ms for sweep 2 down to -162.5 ms for sweep 7.
    let (lo, mean, hi) = (
        t["degraded_pair_age_ms_min"].as_f64(),
        t["degraded_pair_age_ms_mean"].as_f64(),
        t["degraded_pair_age_ms_max"].as_f64(),
    );
    let (Some(lo), Some(mean), Some(hi)) = (lo, mean, hi) else {
        panic!("the degraded sets' pair age is missing: {t}");
    };
    let (stalest, freshest) = (-162.5, -37.5);
    assert!(
        lo >= stalest - 1e-9 && hi <= freshest + 1e-9 && lo <= mean && mean <= hi,
        "degraded pair age {lo} / {mean} / {hi} ms is not frame 1 against a later sweep's trigger: {t}"
    );
    assert!(
        out.lines().any(|l| l.starts_with("pair age DEGRADED = -")),
        "the run never printed how stale its degraded sets were:\n{out}"
    );
    // Its control: a run that degraded nothing has no such age, and no line.
    assert!(
        t0["degraded_pair_age_ms_mean"].is_null(),
        "a run with no degraded set reports a degraded pair age: {t0}"
    );
    assert!(!stdout(&o0).contains("pair age DEGRADED"));

    // 4. the provenance join holds for the degraded sets too, against the
    //    window that produced them. It used to hold them to the completed
    //    sets' rule -- inside the sweep -- which a stale pair breaks by
    //    definition, so every stale run printed a MISMATCH and reported
    //    `cam_pair_join_ok = false` while nothing here looked.
    assert_eq!(
        t["cam_pair_join_ok"],
        serde_json::json!(true),
        "the stale sets failed the provenance join: {t}"
    );
    let join = out
        .lines()
        .find(|l| l.starts_with("fused cam_seq = a frame the camera really admitted"))
        .unwrap_or_else(|| panic!("no join line:\n{out}"));
    assert!(
        join.contains(&format!(
            "at most 1000 ms before its trigger, the declared stale window, on all {degraded} degraded ones"
        )),
        "{join}"
    );
}

/// `--pair-stale-ms` is counted back from the sweep's TRIGGER -- the instant
/// every stale label counts from -- and not from the start of its range, half
/// a period earlier. It used to be the start, and `--pair-stale-ms 50` on
/// drive_0005 produced sets labelled "stale 93 ms".
///
/// Deterministic on the stale-window fixture: two frames, eight sweeps, 25 ms
/// apart, each trigger half a period into its sweep. Sweep j's newest older
/// frame is frame 1, `(j - 1) * 25` ms before its START but 12.5 ms more
/// before its TRIGGER. A 160 ms window therefore admits sweep 7 counted from
/// the start (150 ms) and refuses it counted from the trigger (162.5 ms) --
/// and sweep 7 is the last sweep, which every `DropOldest` edge upstream
/// keeps, so it always reaches the fusion. Counted from the start this run
/// expires nothing; counted from the trigger it expires sweep 7 at least --
/// as `pair_absent_in_source`, because the drive has the lidar's eight frames
/// and the camera's source only two: frames 2-7 are a gap in the camera's
/// source, so a set the window refuses has no frame for that reason, never
/// because one was late or dropped.
#[test]
fn the_stale_window_is_counted_back_from_the_sweep_s_trigger() {
    const FRAMES: usize = 2;
    let fx = FixtureDrive::with_velodyne(FRAMES, N, PACED_PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    let cwd = TempDir::new().unwrap();
    let window_ms = 160.0;
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "trigwin",
            "--track",
            "on",
            "--cap",
            "16",
            "--rate",
            "1.0",
            "--pair-wait-ms",
            "2000",
            "--pair-stale-ms",
            "160",
        ],
    );
    assert_ok(&o);
    let t = track_block(&cwd, "trigwin");
    let expired = u64_of(&t, "expired");
    assert!(
        expired >= 1,
        "sweep 7's frame is 162.5 ms before its trigger and a 160 ms window admitted it: the window is not counted from the trigger: {t}"
    );
    let absent = t["expired_absent_in_source"]
        .as_array()
        .unwrap_or_else(|| panic!("no expired_absent_in_source: {t}"));
    assert_eq!(
        absent.len() as u64,
        expired,
        "a set expired for a reason other than no frame in the window: {t}"
    );
    assert!(absent.contains(&serde_json::json!(7)), "{t}");
    // No degraded set is labelled staler than the window that admitted it.
    if u64_of(&t, "degraded") > 0 {
        let stalest = t["degraded_pair_age_ms_min"]
            .as_f64()
            .unwrap_or_else(|| panic!("degraded sets without a pair age: {t}"));
        assert!(
            stalest >= -window_ms - 1e-9,
            "a degraded set is labelled {stalest} ms stale under a {window_ms} ms window: {t}"
        );
    }
    assert_eq!(
        t["cam_pair_join_ok"],
        serde_json::json!(true),
        "the provenance join disagrees with the window that produced the sets: {t}"
    );
}

/// The chain's shape, at the far end, from the columns the answer carries.
///
/// Not a byte count — the fixture's scene decides that — but the structural
/// claims: the answer is one record per track the fusion emitted, each record
/// its documented size, the nearest-in-path flag on exactly the answers that
/// say they have one, and the far end can still name how many raw returns
/// are behind it.
#[test]
fn the_answer_carries_the_whole_chain_behind_it() {
    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "chain", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let t = track_block(&cwd, "chain");
    let st = t["state"].clone();
    assert_eq!(
        u64_of(&st, "object_bytes"),
        pipes_kitti::state::OBJECT_BYTES as u64,
        "an answer record is meant to be 17 floats"
    );
    // One record per track: the invariant that reads the records at the far
    // end, by name, so a carve-out would be visible.
    let out = stdout(&o);
    let line = out
        .lines()
        .find(|l| l.starts_with("INVARIANT answer "))
        .unwrap_or_else(|| {
            panic!(
                "no INVARIANT answer line:
{out}"
            )
        });
    assert!(line.ends_with("-> OK"), "{line}");
    assert_eq!(u64_of(&st, "down_inconsistent_records"), 0, "{st}");
    assert_eq!(u64_of(&st, "down_flag_mismatch"), 0, "{st}");
    assert_eq!(u64_of(&st, "down_count_mismatch"), 0, "{st}");
    assert_eq!(
        u64_of(&st, "down_flagged"),
        u64_of(&st, "down_with_object"),
        "an answer that has an object flags no record, or one without flags one: {st}"
    );
    // The sample carries every record plus provenance: at least the records.
    let carried = u64_of(&st, "out_payload_bytes_per_sample");
    let objects = st["objects_per_answer"].as_f64().unwrap_or(-1.0);
    assert!(
        objects >= 0.0,
        "the far end did not count the records: {st}"
    );
    assert!(
        carried as f64 >= objects * pipes_kitti::state::OBJECT_BYTES as f64,
        "{carried} B carried for {objects} records a sample: {st}"
    );
    // And the far end can state the whole reduction from the sample in hand.
    assert!(
        st["chain_returns_per_answer"].as_f64().unwrap_or(0.0) > 0.0,
        "the far end could not recompute the chain: {st}"
    );
    // The accounting from the fusion to the far end, one link at a time.
    // `track->state` and `state->sink` are cap-4 DropOldest edges
    // (`run::CHAIN_CAP`), so `down_delivered == track.produced` straight across
    // them asserts the scheduler, and it failed that way two runs in thirty.
    // What holds is conservation at each link, with the evictions read from
    // the run's own record -- plus a far end that saw SOMETHING, or the three
    // identities would be satisfied by a run that lost everything.
    let s = summary(&cwd, "chain");
    let (to_state, to_sink) = (
        dropped_on(&s, "track->state"),
        dropped_on(&s, "state->sink"),
    );
    assert_eq!(
        u64_of(&st, "delivered") + to_state,
        u64_of(&t, "produced"),
        "a fused set neither reached `state` nor was counted as evicted: {t}"
    );
    assert_eq!(
        u64_of(&st, "produced") + u64_of(&st, "errors"),
        u64_of(&st, "delivered"),
        "a set `state` consumed became neither an answer nor an error: {st}"
    );
    assert_eq!(
        u64_of(&st, "down_delivered") + to_sink,
        u64_of(&st, "produced"),
        "an answer neither reached the far end nor was counted as evicted: {st}"
    );
    assert!(
        u64_of(&st, "down_delivered") > 0,
        "no answer reached the far end at all: {st}"
    );
}

/// Under `--detector off` the camera half carries no detections, so the
/// fusion associates nothing: no detection is camera-only, the run says so in
/// words rather than printing zeros as though it had looked, and the fusion's
/// invariant holds over nothing. The sets are still written to
/// `fused.arrows`, one per set produced, in the track format, so a run
/// without the detector is inspectable the same way.
///
/// This fixture's scene is a sparse diagonal line that never forms a track,
/// so the per-TRACK claim -- every record unfused, not lidar-only, which would
/// say the camera looked -- is checked here only over whatever tracks exist,
/// and pinned where tracks can be built on purpose: `pipes_kitti::track`'s
/// `build_fused` unit tests.
#[test]
fn without_a_detector_every_track_is_unfused_and_the_sets_are_written() {
    use arrow::ipc::reader::StreamReader;
    use pipes_kitti::fuse::{camera_only_rows, POPULATION_UNFUSED};
    use pipes_kitti::track::{track_format, track_population, track_rows, TRACK_FORMAT};

    let fx = fused_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "nodet", "--track", "on", "--cap", "16"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert!(
        out.lines()
            .any(|l| l.starts_with("fusion: no camera detector (--detector off)")),
        "the run did not say the fusion had nothing to look at:\n{out}"
    );
    assert_line(
        &o,
        "INVARIANT fusion detections=0 fused=0 camera_only=0 column_mismatch=0 down_fused_mismatch=0 -> OK",
    );
    let t = track_block(&cwd, "nodet");
    let f = t["fusion"].clone();
    assert_eq!(f["detector"], serde_json::json!(false), "{f}");
    assert!(
        f["rule"].is_null(),
        "a rule was recorded with no detector: {f}"
    );
    assert_eq!(u64_of(&f, "down_fused"), 0, "{f}");
    assert_eq!(u64_of(&f, "down_lidar_only"), 0, "{f}");
    assert_eq!(u64_of(&f, "down_camera_only"), 0, "{f}");
    // Every record the far end saw is unfused, and there were some.
    let answer = out
        .lines()
        .find(|l| l.starts_with("INVARIANT answer "))
        .unwrap_or_else(|| panic!("no INVARIANT answer line:\n{out}"));
    let records = common::field(answer, "records=") as u64;
    assert_eq!(u64_of(&f, "down_unfused"), records, "{f}");

    // The sets, as written: one per set produced, the track format, every
    // track unfused and no camera-only row.
    let path = common::run_dir(&cwd, "nodet").join("fused.arrows");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut sets = 0u64;
    let mut tracks = 0usize;
    for b in StreamReader::try_new(file, None).expect("an Arrow IPC stream") {
        let b = b.expect("a batch");
        assert_eq!(track_format(&b), Some(TRACK_FORMAT));
        for r in track_rows(&b) {
            assert_eq!(track_population(r), POPULATION_UNFUSED, "{r:?}");
            assert_eq!(r[21], -1.0, "an unfused track names a detection");
        }
        tracks += track_rows(&b).len();
        assert!(camera_only_rows(&b).is_empty());
        sets += 1;
    }
    assert_eq!(
        sets,
        u64_of(&t, "produced"),
        "fused.arrows against the summary"
    );
    assert!(sets > 0, "nothing was produced, so no set was checked: {t}");
    // The far end sees a subset of the sets -- two cap-4 DropOldest edges lie
    // between -- so it can have fewer records than the sets, never more.
    assert!(
        records <= tracks as u64,
        "the far end saw {records} records from sets holding {tracks} tracks"
    );
}
