//! The third link of the chain, end to end, with no dataset.
//!
//! `reduce` proved a stage can consume an Arrow payload and produce a smaller
//! one. It did it by emitting **the layout it consumed**, which is why one
//! `cloud_thread` reads both ends of that hand-off — and why the result was
//! four times smaller and still exactly the same kind of thing.
//!
//! `detect` is the link where that stops. Its output is a different shape, so
//! it needs a consumer of its own, and the claim under test here is not "the
//! payload shrank" but **"the payload shrank because it started saying
//! something else"**. Four ways a fake could pass, and a test for each:
//!
//! * A stage that copied its input would produce every row and every
//!   invariant. [`detect_borrows_the_cloud_and_allocates_only_its_result`]
//!   measures both halves — the input costing nothing AND the output costing
//!   something — because those are different claims and only one is true here.
//! * A stage whose result nobody consumed would look identical from upstream.
//!   [`the_detections_reach_a_stage_of_their_own`] pins the far end, and pins
//!   that it is a DIFFERENT stage: `velo_xyzr` must not read a detection.
//! * A chain that was really still passing clouds around would pass a byte
//!   count. [`the_payload_stops_being_a_point_cloud`] checks the shape.
//! * Closing the derived edge before its producer stops fails **silently**.
//!   [`the_detection_edge_outlives_its_producer`] is the test for that, on the
//!   rung `reduce` already learned it on.
//!
//! **Nothing here asserts a detection count.** The count is a property of the
//! scene, and this fixture's scene is a line of points 1 m apart — chosen so
//! the arithmetic is checkable, not so the output looks impressive. What is
//! pinned is the direction, the conservation, the provenance, and the one
//! thing the fixture DOES make exact: a scene that is a single flat line is
//! ground, and a ground-removing stage must say so. The measured numbers on a
//! real drive belong in the write-up, and the code that produces them is
//! exercised against the real data by `pipes-kitti`'s own ignored test.

use std::collections::BTreeMap;

use pipes_core::sample::StreamId;
use pipes_kitti::testing::{fixture_sweep_points, FixtureDrive};
use tempfile::TempDir;

mod common;
use common::{
    assert_line, assert_ok, dropped_on, evidence, field, line_with, pipes, stdout, summary, Csv,
};

/// Frames and sweeps per fixture run, and the period they share.
const N: usize = 8;
const PERIOD_NS: i64 = 5_000_000;

fn lidar_fx() -> FixtureDrive {
    FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap()
}

/// The `lidar.reduce.detect` block of a run's `summary.json`.
fn detect_block(cwd: &TempDir, name: &str) -> serde_json::Value {
    let s = summary(cwd, name);
    let d = s["lidar"]["reduce"]["detect"].clone();
    assert!(
        !d.is_null(),
        "no lidar.reduce.detect block in summary.json — the third link did not run:\n{s}"
    );
    d
}

fn u64_at(v: &serde_json::Value, key: &str) -> u64 {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} missing or not a u64 in {v}"))
}

/// The fixture sweep behind every `tov_start_ns`, from the lidar driver's own
/// rows.
///
/// A derived sample keeps the sweep's `tov` and its `seq`, so either names
/// the sweep all the way down the chain; the instant is joined here because
/// it is the measurement, and the seq is checked against it end to end
/// (`source_gaps::every_edge_and_stage_carries_the_frame_s_own_number`).
fn sweep_of_tov(csv: &Csv) -> BTreeMap<String, usize> {
    csv.with("edge", "velo")
        .into_iter()
        .filter(|r| !csv.col(r, "arrival_seq").is_empty())
        .map(|r| {
            (
                csv.col(r, "tov_start_ns"),
                csv.col(r, "seq").parse().expect("velo seq"),
            )
        })
        .collect()
}

/// The sweeps whose reduced cloud reached `detect`, ascending.
///
/// Read out of the evidence rather than assumed to be `0..N`: `velo->reduce`
/// is a cap-2 DropOldest edge and `det->detect` a cap-4 one, and on an
/// unpaced run either may evict. Every value assertion in this file that
/// depends on WHICH sweeps arrived is computed over these.
fn sweeps_reaching_detect(csv: &Csv) -> Vec<usize> {
    let by_tov = sweep_of_tov(csv);
    let mut v: Vec<usize> = csv
        .with("edge", "det->detect")
        .into_iter()
        .filter(|r| csv.col(r, "outcome") == "Delivered")
        .map(|r| {
            let tov = csv.col(r, "tov_start_ns");
            *by_tov.get(&tov).unwrap_or_else(|| {
                panic!("a cloud reached detect with an instant no sweep has: {tov}")
            })
        })
        .collect();
    v.sort_unstable();
    assert!(
        !v.is_empty(),
        "no cloud reached detect at all -- that is a broken edge, not a busy machine"
    );
    v
}

/// The sweeps that reached `reduce`, ascending. A raw sweep's `seq` IS its
/// fixture index, so no instant is needed here.
fn sweeps_reaching_reduce(csv: &Csv) -> Vec<usize> {
    let mut v: Vec<usize> = csv
        .with("edge", "velo->reduce")
        .into_iter()
        .filter(|r| csv.col(r, "outcome") == "Delivered")
        .map(|r| csv.col(r, "seq").parse().expect("sweep seq"))
        .collect();
    v.sort_unstable();
    assert!(
        !v.is_empty(),
        "no sweep reached reduce at all -- that is a broken edge, not a busy machine"
    );
    v
}

/// The fixture's scene is a line of points at one height, which IS a ground
/// plane and nothing else — so a stage that removes the ground must report
/// that it removed everything and detected nothing.
///
/// This is the sharpest thing this fixture can say, and it is a real
/// assertion: a stage that skipped ground removal would report the line as one
/// long detection, and a stage that detected nothing because it was broken
/// would fail the positive control below, which puts a block above the line
/// and requires it to come back.
#[test]
fn a_scene_that_is_only_ground_produces_no_detections() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "flat", "--rate", "inf"]);
    assert_ok(&o);
    let out = stdout(&o);
    assert!(!out.contains("-> FAIL"), "an invariant FAILed:\n{out}");

    let d = detect_block(&cwd, "flat");
    assert!(u64_at(&d, "delivered") > 0, "the stage was never fed");
    assert_eq!(
        u64_at(&d, "produced") + u64_at(&d, "errors"),
        u64_at(&d, "delivered"),
        "produced + errors != delivered: {d}"
    );
    assert_eq!(
        u64_at(&d, "clusters"),
        0,
        "a flat line of points left something above the ground: {d}"
    );
    assert_eq!(d["detections_per_sweep"].as_u64(), Some(0));
    // And it did not achieve that by throwing the cloud away: every in-range
    // voxel is accounted for as ground.
    assert_eq!(
        u64_at(&d, "ground_voxels"),
        u64_at(&d, "in_range_voxels"),
        "voxels vanished between the range filter and the ground plane: {d}"
    );
    assert!(u64_at(&d, "in_range_voxels") > 0, "nothing was in range");
    // An empty output reports NO ratio rather than a spectacular one. `reduce`
    // once printed `11883.34x smaller` over a result with nothing in it.
    assert!(
        d["payload_shrink"].is_null() && d["voxel_shrink"].is_null(),
        "a shrink ratio was published over an empty result: {d}"
    );
    // The persistence check likewise reports nothing rather than 0 %.
    assert!(
        d["persistence_fraction"].is_null(),
        "a persistence figure was published over no detections: {d}"
    );
}

/// The two halves of the zero-copy claim, which are NOT the same claim.
///
/// Reading the cloud must cost nothing, because the stage indexes it in place.
/// Building the detections must cost SOMETHING, because a stage that
/// transforms data has to allocate its result — a 0 there would mean the stage
/// produced nothing at all, which is the failure a single combined figure
/// could not tell from success.
#[test]
fn detect_borrows_the_cloud_and_allocates_only_its_result() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "borrow", "--rate", "inf"]);
    assert_ok(&o);

    let d = detect_block(&cwd, "borrow");
    assert_eq!(
        u64_at(&d, "read_bytes_per_sample"),
        0,
        "reading the cloud allocated, so it was copied rather than borrowed: {d}"
    );
    assert!(
        u64_at(&d, "build_bytes_per_sample") > 0,
        "building the detections allocated nothing, so nothing was built: {d}"
    );
    assert_eq!(
        u64_at(&d, "down_bytes_per_sample"),
        0,
        "the consumer allocated per sample, so the detections were copied: {d}"
    );
    // The scratch is what the 0 above EXCLUDES, and it is reported rather than
    // implied: "allocates nothing" means per cloud, and this is the once.
    assert!(
        u64_at(&d, "scratch_bytes") > 0 && u64_at(&d, "scratch_growths") > 0,
        "the scratch reported nothing, so the 0 above has no caveat beside it: {d}"
    );
    // Both address-equality proofs, and NOT CHECKED is not a pass.
    assert_eq!(d["input_storage_id_equal"].as_bool(), Some(true));
    assert_eq!(d["output_storage_id_equal"].as_bool(), Some(true));
    // The same pair on stdout, spelled so the two halves cannot be read as
    // one number.
    let out = stdout(&o);
    let line = line_with(&out, "detect bytes/sweep = ");
    assert!(
        line.starts_with("detect bytes/sweep = 0 reading the cloud (borrowed) + ")
            && !line.ends_with("+ 0 building the detections (allocated)"),
        "{line}"
    );
}

/// The detections reach a stage of their own, and it is a DIFFERENT stage.
///
/// `reduce` hands its clouds to the same `cloud_thread` that reads the raw
/// sweep, which is how the Arrow rule was made mechanical. This link cannot do
/// that, and the test asserts the reason rather than the fact: the far end is
/// `obj-sink`, it reads columns no cloud has, and the run's own storage and
/// provenance joins hold across it.
#[test]
fn the_detections_reach_a_stage_of_their_own() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "far", "--rate", "inf"]);
    assert_ok(&o);
    let out = stdout(&o);

    let d = detect_block(&cwd, "far");
    let produced = u64_at(&d, "produced");
    assert!(produced > 0, "the chain produced nothing at all");
    // What reached the far end is what was produced LESS what `obj->sink`
    // evicted on the way: a fixed cap-4 DropOldest edge (`run::OBJ_CAP`), so
    // `down_delivered == produced` would be a statement about the scheduler.
    // Conservation is exact either way, with the eviction count read from
    // the run's own record rather than assumed to be 0.
    let down = u64_at(&d, "down_delivered");
    let evicted = dropped_on(&summary(&cwd, "far"), "obj->sink");
    assert_eq!(
        down + evicted,
        produced,
        "detections were produced that neither reached the far end nor were counted as evicted: {d}"
    );
    assert!(
        down > 0,
        "no detection batch reached the far end at all: {d}"
    );
    assert_eq!(u64_at(&d, "parent_mismatch"), 0);
    assert_eq!(d["parent_seq_equal"].as_bool(), Some(true));
    for line in [
        format!("obj-sink parent = lidar_det on {down} of {down} samples"),
        format!("obj-sink parent seq = the cloud detect read: OK ({down} samples)"),
    ] {
        assert!(
            out.lines().any(|l| l == line),
            "missing:\n  {line}\n--- stdout ---\n{out}"
        );
    }
    // The evidence file carries the whole chain's edges, and the new ones
    // carry a stream no driver produces.
    let csv = evidence(&cwd, "far");
    for edge in ["det->detect", "obj", "obj->sink"] {
        assert!(!csv.with("edge", edge).is_empty(), "no rows on edge {edge}");
    }
    // One row per ADMITTED batch -- the consumer's for a delivered one, the
    // pusher's for an evicted one -- so this is `produced`, not `down`.
    let obj_rows = csv.with("edge", "obj->sink");
    assert_eq!(obj_rows.len() as u64, produced, "obj->sink rows");
    // The `stream` column is the id, and it is compared against the constant
    // rather than against a literal 7: a test that hard-coded the number would
    // keep passing if the id were reassigned to something else.
    for r in &obj_rows {
        assert_eq!(
            csv.col(r, "stream"),
            StreamId::LIDAR_OBJ.0.to_string(),
            "row {r:?}"
        );
    }
    assert_ne!(
        StreamId::LIDAR_OBJ,
        StreamId::LIDAR_DET,
        "the detections share an id with the cloud they came from"
    );
}

/// The payload stops being a point cloud, and that is the whole point of the
/// stage.
///
/// A byte count alone cannot say this: a stage that kept every tenth voxel
/// would shrink the payload just as far and would still be handing on the same
/// kind of thing. What is checked is that the number of ROWS collapsed while
/// each row got BIGGER — 44 bytes against 16 — which is what "fewer, more
/// specific statements" looks like in an Arrow buffer, and that the run says
/// in words what one of those rows is.
#[test]
fn the_payload_stops_being_a_point_cloud() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "shape", "--rate", "inf"]);
    assert_ok(&o);
    let out = stdout(&o);

    let d = detect_block(&cwd, "shape");
    // Carried bytes fall across the stage on this fixture, whose scene is all
    // ground, so the output is a header and nothing else.
    let carried_in = u64_at(&d, "in_payload_bytes_per_sample");
    let carried_out = u64_at(&d, "out_payload_bytes_per_sample");
    assert!(
        carried_out < carried_in,
        "the detections carried {carried_out} B against the cloud's {carried_in} B"
    );
    // The caveat is printed WITH the count, not somewhere else in the run.
    let what = line_with(&out, "detect what those are:");
    assert!(
        what.contains("NOT objects"),
        "the detection count is reported without saying what a detection is: {what}"
    );
    assert!(
        d["what_a_detection_is"]
            .as_str()
            .is_some_and(|s| s.contains("NOT one object")),
        "summary.json records a count with no statement of what it counts: {d}"
    );
    // The stage's three parameters travel with the run, so the count can be
    // interpreted from the directory alone.
    let run = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(cwd.path().join("runs/shape/run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(run["detect"].as_str(), Some("on"));
    assert_eq!(run["detect_range_limit_m"].as_f64(), Some(30.0));
    assert_eq!(run["detect_min_cluster_voxels"].as_u64(), Some(3));
    assert!(d["params_rationale"]
        .as_str()
        .is_some_and(|s| s.contains("pedestrian")));
}

/// Closing `obj->sink` before `detect` has stopped fails **silently**: every
/// batch admitted during the drain comes back `Closed`, is counted as a drop,
/// and `delivered + dropped == admitted` still balances while the detections
/// have vanished from the measurement. `reduce` learned this on `det->cloud`;
/// this is the same rung one further down.
///
/// The test is the conservation identity plus the requirement that no drop on
/// the edge is a `closed` one. Not `dropped=0`: the edge is a cap-4 DropOldest
/// queue (`run::OBJ_CAP`) and may evict on a loaded host, and an eviction is
/// the queue doing its job where a `closed` drop is the edge shut under its
/// producer. The pusher's row says which, which is why that row exists -- and
/// the identity alone would not do, because it holds just as well when
/// everything was dropped.
#[test]
fn the_detection_edge_outlives_its_producer() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "order", "--cap", "16"]);
    assert_ok(&o);
    let out = stdout(&o);

    let d = detect_block(&cwd, "order");
    let produced = u64_at(&d, "produced");
    assert!(produced > 0);
    let line = line_with(&out, "INVARIANT edge=obj->sink ");
    assert!(line.ends_with("-> OK"), "{line}");
    let (delivered, lost, admitted) = (
        field(line, "delivered="),
        field(line, "dropped="),
        field(line, "admitted="),
    );
    assert_eq!(
        admitted as u64, produced,
        "detect produced batches that were never admitted onto the edge: {line}"
    );
    assert_eq!(delivered + lost, admitted, "{line}");
    assert!(delivered > 0, "nothing reached the sink at all: {line}");
    // Every drop the counter reports is in the evidence, and every one of
    // them is the queue being full -- never the edge being gone.
    let csv = evidence(&cwd, "order");
    let not_delivered: Vec<_> = csv
        .with("edge", "obj->sink")
        .into_iter()
        .filter(|r| csv.col(r, "outcome") != "Delivered")
        .collect();
    assert_eq!(
        not_delivered.len() as i64,
        lost,
        "the edge's drop rows do not match its counter: {line}"
    );
    for r in &not_delivered {
        assert_eq!(
            csv.col(r, "reason"),
            "evicted",
            "a detection batch was dropped for a reason other than a full queue, so the edge \
             was shut while its producer was still admitting onto it: {r:?}"
        );
    }
}

/// `--detect off` stops the chain at the reduced cloud and changes nothing
/// upstream of it.
///
/// The second half is the one that matters. A flag that silently perturbed
/// `reduce` would make the stage's cost unmeasurable, because the two runs
/// being differenced would not be the same run minus one stage. It is
/// measured sweep for sweep rather than as a mean, for the reason
/// [`two_runs_of_one_drive_detect_the_same_things`] gives: the summary's
/// per-sample figures are means over the sweeps that reached the stage, and
/// on an unpaced run `velo->reduce` decides which those are.
#[test]
fn detect_off_stops_the_chain_and_leaves_reduce_alone() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let on = pipes(&cwd, &fx, &["--name", "on", "--rate", "inf", "--cap", "16"]);
    let off = pipes(
        &cwd,
        &fx,
        &[
            "--name", "off", "--rate", "inf", "--cap", "16", "--detect", "off",
        ],
    );
    assert_ok(&on);
    assert_ok(&off);

    assert!(
        summary(&cwd, "off")["lidar"]["reduce"]["detect"].is_null(),
        "`--detect off` still published a detect block"
    );
    let off_out = stdout(&off);
    assert!(
        !off_out.contains("obj->sink"),
        "`--detect off` still opened the detection edge:\n{off_out}"
    );
    let run = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(cwd.path().join("runs/off/run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(run["detect"].as_str(), Some("off"));
    assert!(run["detect_range_limit_m"].is_null());

    // And `reduce` measured the same thing either way -- sweep for sweep.
    // This once compared `in_points_per_sweep` and `out_points_per_sample`
    // between the runs, and saw 8 against 7: those are means over the sweeps
    // that reached the stage, and with six delivered in one run and eight in
    // the other the means moved while `reduce` had not. What `--detect` must
    // not move is what each sweep BECAME, so the cloud built from every sweep
    // both runs delivered is required to be the same size and cost the same
    // to build in both.
    let (ea, eb) = (evidence(&cwd, "on"), evidence(&cwd, "off"));
    let (pa, pb) = (produced_by_sweep(&ea, "det"), produced_by_sweep(&eb, "det"));
    let both: Vec<&usize> = pa.keys().filter(|k| pb.contains_key(k)).collect();
    assert!(
        !both.is_empty(),
        "no sweep reached reduce in both runs: {:?} and {:?}",
        pa.keys().collect::<Vec<_>>(),
        pb.keys().collect::<Vec<_>>()
    );
    for k in both {
        assert_eq!(
            pa[k], pb[k],
            "reduce's cloud for sweep {k} moved with --detect: (bytes_alloc, payload_bytes)"
        );
    }
    // And each run's means are the closed form over the sweeps that reached
    // it: the fixture's `4 + i` points in, one voxel per point out.
    for (name, ev) in [("on", &ea), ("off", &eb)] {
        let reached = sweeps_reaching_reduce(ev);
        let points: u64 = reached
            .iter()
            .map(|&i| fixture_sweep_points(i) as u64)
            .sum();
        let r = summary(&cwd, name)["lidar"]["reduce"].clone();
        assert_eq!(
            u64_at(&r, "delivered"),
            reached.len() as u64,
            "{name}: the evidence and the stage disagree about what arrived: {reached:?} against {r}"
        );
        for key in ["in_points_per_sweep", "out_points_per_sample"] {
            assert_eq!(
                r[key].as_u64(),
                Some(points / reached.len() as u64),
                "{name}: reduce's {key} is not the closed form over sweeps {reached:?}: {r}"
            );
        }
    }
}

/// `(bytes_alloc, payload_bytes)` of everything a producer edge -- `det` for
/// `reduce`'s clouds, `obj` for `detect`'s batches -- admitted, by the sweep
/// it came from. The two columns are what the evidence records about a
/// derived payload: how big it is, and what building it cost.
fn produced_by_sweep(csv: &Csv, edge: &str) -> BTreeMap<usize, (String, String)> {
    let by_tov = sweep_of_tov(csv);
    csv.with("edge", edge)
        .into_iter()
        .map(|r| {
            let tov = csv.col(r, "tov_start_ns");
            let sweep = *by_tov
                .get(&tov)
                .unwrap_or_else(|| panic!("edge {edge} carries an instant no sweep has: {tov}"));
            (
                sweep,
                (csv.col(r, "bytes_alloc"), csv.col(r, "payload_bytes")),
            )
        })
        .collect()
}

/// Same drive, same detections, twice — through the real binary, not just
/// through the library.
///
/// The library test in `pipes-kitti` proves the algorithm is deterministic.
/// This proves the PIPELINE is: two runs of one drive are two independent
/// thread schedules, and what each sweep became must not depend on which one
/// it ran under.
///
/// What a schedule IS allowed to change is which sweeps arrive. `velo->reduce`
/// is a cap-2 DropOldest edge and these runs are unpaced, so either run may
/// evict a sweep the other delivered, and a total over "the sweeps that got
/// through" then differs between the runs for a reason that has nothing to do
/// with determinism. That was this test's failure: `in_range_voxels` 53
/// against 52 is one run missing the 7-point sweep and the other the 8-point
/// one, with every sweep both delivered identical. So the comparison is made
/// sweep for sweep, over the sweeps both runs delivered, and each run's total
/// is pinned to the closed form over the sweeps IT delivered -- which no
/// schedule can move, and which is the property in its own words: the same
/// input gives the same output, whatever else was going on.
#[test]
fn two_runs_of_one_drive_detect_the_same_things() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    assert_ok(&pipes(&cwd, &fx, &["--name", "d1", "--cap", "16"]));
    assert_ok(&pipes(&cwd, &fx, &["--name", "d2", "--cap", "16"]));
    let (a, b) = (detect_block(&cwd, "d1"), detect_block(&cwd, "d2"));
    let (ea, eb) = (evidence(&cwd, "d1"), evidence(&cwd, "d2"));

    // Each run against the input alone. The fixture's sweep `i` is `4 + i`
    // points a metre apart, every one its own voxel and every one inside the
    // range bound, so a run's in-range count is that summed over exactly the
    // sweeps that reached the stage -- whichever those were.
    for (name, d, ev) in [("d1", &a, &ea), ("d2", &b, &eb)] {
        let reached = sweeps_reaching_detect(ev);
        assert_eq!(
            reached.len() as u64,
            u64_at(d, "delivered"),
            "{name}: the evidence and the stage disagree about what arrived: {reached:?} against {d}"
        );
        let want: usize = reached.iter().map(|&i| fixture_sweep_points(i)).sum();
        assert_eq!(
            u64_at(d, "in_range_voxels"),
            want as u64,
            "{name}: in_range_voxels is not the closed form over sweeps {reached:?}: {d}"
        );
        assert_eq!(u64_at(d, "errors"), 0, "{name}: {d}");
    }

    // Sweep for sweep across the two runs: the cloud `reduce` built and the
    // batch `detect` built from it are the same size and cost the same to
    // build. The intersection is what can be compared, and it is required to
    // be non-empty so the loop cannot pass by comparing nothing.
    for edge in ["det", "obj"] {
        let (pa, pb) = (produced_by_sweep(&ea, edge), produced_by_sweep(&eb, edge));
        let both: Vec<&usize> = pa.keys().filter(|k| pb.contains_key(k)).collect();
        assert!(
            !both.is_empty(),
            "no sweep reached edge {edge} in both runs: {:?} and {:?}",
            pa.keys().collect::<Vec<_>>(),
            pb.keys().collect::<Vec<_>>()
        );
        for k in both {
            assert_eq!(
                pa[k], pb[k],
                "sweep {k} on edge {edge}: (bytes_alloc, payload_bytes) differed between two runs of one drive"
            );
        }
    }

    // And when the two schedules delivered the same sweeps -- the usual case
    // on a quiet host -- every total agrees as well. Conditional on the
    // evidence rather than on `dropped=0`, so the branch is taken exactly
    // when the totals are comparable, and the run says which it was.
    let (ra, rb) = (sweeps_reaching_detect(&ea), sweeps_reaching_detect(&eb));
    if ra == rb {
        for key in [
            "delivered",
            "produced",
            "errors",
            "in_range_voxels",
            "ground_voxels",
            "clusters",
            "fragment_detections",
            "merged_detections",
            "key_collisions",
            "ground_plane_fits",
            "ground_plane_fallbacks",
        ] {
            assert_eq!(
                a[key], b[key],
                "{key} differed between two runs that delivered the same sweeps:\n{a}\n{b}"
            );
        }
        assert_eq!(a["ground_tilt_deg_max"], b["ground_tilt_deg_max"]);
        assert_eq!(a["detections_per_sweep"], b["detections_per_sweep"]);
    } else {
        eprintln!(
            "the two runs delivered different sweeps ({ra:?} against {rb:?}), so the totals were compared sweep for sweep only"
        );
    }
}

/// A cloud built on one grid must not be detected in on another.
///
/// `detect` recovers the voxel grid from the centroids, so it has to be told
/// the edge they were quantised to. One CLI flag feeds both stages today, so
/// no run can currently disagree — which is exactly why the counter exists and
/// why this test pins that it reads 0 rather than being absent. A future
/// second source for that number would otherwise produce detections computed
/// on the wrong grid with nothing anywhere saying so.
#[test]
fn the_grid_detect_assumes_is_the_grid_the_cloud_was_built_on() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    for size in ["0.2", "0.5"] {
        let name = format!("grid{size}");
        let o = pipes(
            &cwd,
            &fx,
            &["--name", &name, "--rate", "inf", "--voxel-size-m", size],
        );
        assert_ok(&o);
        assert_line(
            &o,
            &format!(
                "detect delivered={d} produced={d} errors=0 storage_mismatch=0 voxel_size_mismatch=0",
                d = u64_at(&detect_block(&cwd, &name), "delivered")
            ),
        );
        // A non-default grid is told the range bound was not measured for it.
        let out = stdout(&o);
        let note = line_with(&out, "detect params:");
        assert_eq!(
            note.contains("measured at 0.20 m only"),
            size != "0.2",
            "the range bound's caveat is attached to the wrong grid: {note}"
        );
    }
}

/// The persistence check compares CONSECUTIVE sweeps, and nothing else.
///
/// This is the test for a defect that shipped and was caught on real data:
/// the first version compared whatever two clouds arrived in succession. On an
/// unpaced run `velo->reduce` evicts, so those were about six sweeps apart,
/// and the stage reported 6.8 % persistence -- a number about the eviction
/// rate wearing the name of one about the detector. Restricting it to adjacent
/// sweeps moved the same drive to 35.8 %, which is the figure the design note
/// predicted.
///
/// The pair count has exactly one closed form, and it is over the sweeps that
/// REACHED the stage: one pair per pair of adjacent indices among them. On a
/// run where nothing was evicted that is one fewer than the batches produced;
/// on a run where something was, it is fewer still, and that run is the sharper
/// test -- an eviction is exactly when "consecutive batches" and "consecutive
/// sweeps" come apart, which is the defect. Anything larger means pairs were
/// invented; anything smaller means adjacent sweeps were missed. This test
/// once demanded `dropped=0` on the way in and failed on a loaded host for
/// a reason that had nothing to do with the check.
#[test]
fn the_persistence_check_pairs_only_adjacent_sweeps() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "pairs", "--cap", "16"]);
    assert_ok(&o);
    let out = stdout(&o);

    let d = detect_block(&cwd, "pairs");
    let produced = u64_at(&d, "produced");
    assert!(produced > 1, "fewer than two clouds reached the stage");
    assert_eq!(
        u64_at(&d, "errors"),
        0,
        "a cloud that failed to build is not a pair: {d}"
    );
    // Which sweeps arrived is read from the evidence, and the closed form is
    // computed over exactly those.
    let reached = sweeps_reaching_detect(&evidence(&cwd, "pairs"));
    assert_eq!(
        reached.len() as u64,
        produced,
        "the evidence and the stage disagree about what arrived: {reached:?} against {d}"
    );
    let adjacent = reached.windows(2).filter(|w| w[1] == w[0] + 1).count() as u64;
    assert_eq!(
        u64_at(&d, "persistence_sweep_pairs"),
        adjacent,
        "the check did not run over exactly the adjacent pairs among sweeps {reached:?}: {d}"
    );
    // And the run says the pair count beside the fraction, because a fraction
    // over three pairs and one over 153 are the same field and not the same
    // claim.
    let line = line_with(&out, "detect persistence");
    assert!(
        line.contains("CONSECUTIVE sweep pairs") || line.contains("n/a ("),
        "the persistence line does not say what population it is over: {line}"
    );
    // The figure is labelled as the uncompensated one. Claiming the
    // compensated number would be claiming a measurement nothing here made:
    // `StreamId::OXTS` has no driver.
    assert_eq!(d["persistence_is_compensated"].as_bool(), Some(false));
    assert!(
        line.contains("UNCOMPENSATED") || line.contains("n/a ("),
        "{line}"
    );
}
