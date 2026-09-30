//! The stage-to-stage Arrow hand-off, end to end, with no dataset.
//!
//! The claim is the one the whole project is betting on and had never tested:
//! that a stage can **consume an Arrow payload and produce a smaller Arrow
//! payload that another stage consumes**. Everything before M12 was a driver
//! fanning out to leaf consumers, which is a different shape and proves a
//! different thing.
//!
//! Four ways a fake could pass, so each has a test that bites:
//!
//! * A stage that copied its input would produce every row and every
//!   invariant. [`reduce_borrows_the_sweep_and_allocates_only_its_result`]
//!   measures both halves — the input costing nothing AND the output costing
//!   something — because "the input was not copied" and "nothing was
//!   allocated" are different claims and only one of them is true here.
//! * A chain that produced a result nobody consumed would look identical from
//!   upstream. [`the_derived_cloud_reaches_a_second_stage`] pins the far end.
//! * A derived stream that never reached admission would still let `reduce`
//!   report what it built. [`the_derived_stream_is_admitted_like_a_sensors`]
//!   pins the accounting against the global counter.
//! * Closing the derived edge before its producer stops fails **silently** —
//!   the samples become `closed` drop rows, conservation still balances and
//!   every invariant still prints OK.
//!   [`the_derived_edge_outlives_its_producer`] is the test for that, and it
//!   is the sharpest one here.
//!
//! **Nothing here asserts a particular shrink ratio.** The ratio is a property
//! of the scene, and the fixture's scene is a line of points 1 m apart, chosen
//! to make the arithmetic checkable rather than to look impressive. What is
//! pinned is the direction, the conservation, and the closed-form point count
//! the fixture makes predictable — together with the case where the payload
//! legitimately does **not** shrink, because a grid finer than the scene
//! merges nothing. Pinning a number here would be pinning this fixture; the
//! measured ratio on real data belongs in the write-up.

use pipes_kitti::testing::{fixture_sweep_points, FixtureDrive};
use tempfile::TempDir;

mod common;
use common::{assert_line, assert_ok, evidence, line_with, pipes, run_json, stdout, summary, Csv};

/// Frames and sweeps per fixture run, and the period they share.
const N: usize = 8;
const PERIOD_NS: i64 = 5_000_000;

/// Bytes per point, in and out: the reduction preserves the layout, which is
/// the whole point of it.
const POINT_BYTES: u64 = 16;

fn lidar_fx() -> FixtureDrive {
    FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap()
}

/// The `lidar.reduce` block of a run's `summary.json`.
fn reduce_block(cwd: &TempDir, name: &str) -> serde_json::Value {
    let s = summary(cwd, name);
    let r = s["lidar"]["reduce"].clone();
    assert!(
        !r.is_null(),
        "no lidar.reduce block in summary.json — the chain did not run:\n{s}"
    );
    r
}

fn u64_at(v: &serde_json::Value, key: &str) -> u64 {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is {}, not a number, in {v}", v[key]))
}

fn f64_at(v: &serde_json::Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} is {}, not a number, in {v}", v[key]))
}

/// The `seq`s a stage wrote rows for **on one edge**, ascending.
///
/// Stage AND edge, not stage alone: `reduce` writes two rows per sweep — a
/// consumer row on `velo->reduce` and a producer row on `det` — so selecting
/// by stage counts every sweep twice. That is the first thing in this
/// workspace to make "one row per stage per sample" false, and it is worth the
/// extra argument rather than a subtly doubled denominator.
///
/// Every value assertion is computed over these rather than over `0..N`,
/// because the derived edge is a fixed-capacity Drop* one and may legitimately
/// evict on a busy machine — a test that assumed full delivery would fail for
/// a reason that has nothing to do with what it checks.
fn seqs_of(csv: &Csv, stage: &str, edge: &str) -> Vec<usize> {
    let mut v: Vec<usize> = csv
        .with("edge", edge)
        .iter()
        .filter(|r| csv.col(r, "stage") == stage)
        .map(|r| csv.col(r, "seq").parse().expect("seq"))
        .collect();
    v.sort_unstable();
    assert!(
        !v.is_empty(),
        "stage {stage} wrote no rows on {edge} at all — that is a broken edge, \
         not a busy machine"
    );
    v
}

/// **The deliverable**, and its own negative control, because the honest
/// version of this claim is conditional.
///
/// The fixture's sweep `i` is `4 + i` points at `(j, -j, i, 0.5)` — a line
/// with consecutive points **1 m apart**. So:
///
/// * at a grid COARSER than that spacing the points collapse and the payload
///   falls, which is the claim;
/// * at a grid FINER than it nothing collapses, the reduction is the identity,
///   and the derived payload is a few bytes **larger** than its input, because
///   it carries two extra metadata columns (`voxel_size_m`,
///   `source_point_count`).
///
/// Both are asserted. The second is not a caveat grudgingly admitted, it is
/// the point being defended: the shrink ratio is a property of the SCENE, not
/// a dial on the stage. A test that only ever showed the payload falling would
/// be evidence for a claim this project is careful not to make.
///
/// The ratio on real data belongs in the write-up, measured on a real scene.
/// Pinning a number here would be pinning this fixture.
#[test]
fn the_payload_gets_smaller_when_the_scene_is_denser_than_the_grid() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    // 4 m: coarser than the fixture's 1 m point spacing, so points share cells.
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "chain", "--cap", "16", "--voxel-size-m", "4"],
    );
    assert_ok(&o);
    let r = reduce_block(&cwd, "chain");

    let carried_in = u64_at(&r, "in_payload_bytes_per_sweep");
    let carried_out = u64_at(&r, "out_payload_bytes_per_sample");
    assert!(
        carried_out < carried_in,
        "the transfer did not shrink: {carried_in} B in, {carried_out} B out"
    );
    assert!(f64_at(&r, "payload_shrink") > 1.0);
    assert!(
        u64_at(&r, "out_points_per_sample") < u64_at(&r, "in_points_per_sweep"),
        "the bytes fell without the point count falling, so something other \
         than the reduction moved them"
    );
    assert!(f64_at(&r, "point_shrink") > 1.0);
    // Nothing was discarded, so the shrink is the collapse and not a filter.
    assert_eq!(u64_at(&r, "non_finite_points"), 0);
    assert_eq!(u64_at(&r, "out_of_range_points"), 0);

    // Carried bytes really are the points, checked against the fixture's own
    // closed form rather than against the summary that reported them.
    let csv = evidence(&cwd, "chain");
    let reduced = seqs_of(&csv, "reduce", "velo->reduce");
    let in_points: u64 = reduced
        .iter()
        .map(|&i| fixture_sweep_points(i) as u64)
        .sum();
    assert_eq!(
        u64_at(&r, "delivered"),
        reduced.len() as u64,
        "summary and evidence disagree on what reduce consumed"
    );
    assert!(
        u64_at(&r, "in_payload_bytes_total") > in_points * POINT_BYTES,
        "carried bytes do not even cover the points the fixture wrote"
    );

    // The same numbers on stdout, where a reader of a run meets them.
    let out = stdout(&o);
    for prefix in ["byte chain 1 ", "byte chain 2 ", "byte chain shrink = "] {
        assert!(
            out.lines().any(|l| l.starts_with(prefix)),
            "no {prefix:?} line:\n{out}"
        );
    }

    // The control, same fixture, default 0.2 m grid: finer than the scene, so
    // the reduction keeps every point and the payload does NOT fall.
    assert_ok(&pipes(&cwd, &fx, &["--name", "fine", "--cap", "16"]));
    let f = reduce_block(&cwd, "fine");
    assert_eq!(
        u64_at(&f, "out_points_per_sample"),
        u64_at(&f, "in_points_per_sweep"),
        "a 0.2 m grid merged points the fixture wrote 1 m apart"
    );
    assert_eq!(f64_at(&f, "point_shrink"), 1.0);
    assert!(
        u64_at(&f, "out_payload_bytes_total") >= u64_at(&f, "in_payload_bytes_total"),
        "the derived payload got smaller with nothing merged, so the bytes are \
         not the points"
    );
}

/// The measurement the task turns on: the input is BORROWED and the output is
/// ALLOCATED, and those are two different numbers rather than one.
///
/// A stage that copied its 1.95 MB input would still shrink the payload, still
/// balance every invariant and still pass every other test in this file. What
/// catches it is the first assertion; what stops that assertion being vacuous
/// is the second, because a stage that did nothing at all would also report 0.
#[test]
fn reduce_borrows_the_sweep_and_allocates_only_its_result() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "borrow", "--cap", "16"]);
    assert_ok(&o);
    let r = reduce_block(&cwd, "borrow");

    // Reading the sweep cost nothing: it is a slice into the driver's buffer.
    assert_eq!(
        u64_at(&r, "read_bytes_per_sweep"),
        0,
        "reduce allocated while READING the sweep, so it copied its input"
    );
    // Positive control, in the same test: building the result cost something.
    // Without this the assertion above would pass against a stage that never
    // ran, and "the input was not copied" would be a statement about nothing.
    assert!(
        u64_at(&r, "build_bytes_per_sample") > 0,
        "reduce allocated nothing building its output either, so it produced nothing"
    );
    // And the far end allocates nothing, exactly as the raw cloud's consumer
    // does — the derived buffer is shared onward, not copied again.
    assert_eq!(u64_at(&r, "down_bytes_per_sample"), 0);

    // The address proofs, both links. These are what say "borrowed" rather
    // than "cheap": the buffer `reduce` read is the one the driver allocated,
    // and the buffer the far consumer read is the one `reduce` allocated.
    assert_eq!(r["input_storage_id_equal"], serde_json::json!(true));
    assert_eq!(r["output_storage_id_equal"], serde_json::json!(true));
    // The count in that line is what `reduce` CONSUMED, not `N`. `velo->reduce`
    // is a fixed cap-2 Drop* edge and these runs are `--rate inf`, so being
    // outrun by the driver is correct behaviour and happened about one run in
    // seven -- this assertion failed for a reason that had nothing to do with
    // borrowing. `> 0` keeps it from passing against a stage that ran zero
    // sweeps and reported OK over an empty population.
    let delivered = u64_at(&r, "delivered");
    let produced = u64_at(&r, "produced");
    assert!(delivered > 0 && produced > 0, "reduce consumed nothing");
    assert_line(
        &o,
        &format!("reduce read the driver's buffer: OK ({delivered} sweeps)"),
    );

    // The caveat on that 0, published as a number rather than implied: the
    // scratch is real, it is just not per sweep.
    assert!(u64_at(&r, "scratch_bytes") > 0);
    assert!(u64_at(&r, "scratch_growths") > 0);

    // The evidence carries the same split per row: the consumer row on
    // `velo->reduce` reads 0, the producer row on `det` does not. Summing the
    // column over the run therefore still totals the stage's allocations once.
    let csv = evidence(&cwd, "borrow");
    let bytes_on = |edge: &str| -> Vec<u64> {
        csv.with("edge", edge)
            .iter()
            .filter(|r| csv.col(r, "stage") == "reduce")
            .map(|r| csv.col(r, "bytes_alloc").parse::<u64>().expect("bytes"))
            .collect()
    };
    let read = bytes_on("velo->reduce");
    let built = bytes_on("det");
    assert_eq!(
        read.len() as u64,
        delivered,
        "one consumer row per sweep CONSUMED"
    );
    assert_eq!(
        built.len() as u64,
        produced,
        "one producer row per derived cloud"
    );
    assert!(
        read.iter().all(|&b| b == 0),
        "a reduce input row allocated: {read:?}"
    );
    assert!(
        built.iter().all(|&b| b > 0),
        "a reduce output row allocated nothing: {built:?}"
    );
}

/// The hand-off itself: a second stage consumes what the first produced, and
/// it is the SAME consumer code that reads the raw sweep.
///
/// That last part is the Arrow rule as the architecture document states it —
/// "processing, model, and fusion stages consume and produce the same
/// combination". If `reduce` emitted a different layout, this run would need a
/// second consumer implementation, and the property would have quietly stopped
/// being true while every byte figure stayed correct.
#[test]
fn the_derived_cloud_reaches_a_second_stage() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "handoff", "--cap", "16"]);
    assert_ok(&o);
    let csv = evidence(&cwd, "handoff");

    // Both ends exist and are on their own stream.
    let far = seqs_of(&csv, "det-cloud", "det->cloud");
    assert!(!far.is_empty());
    for (edge, stream) in [("velo->reduce", "1"), ("det", "3"), ("det->cloud", "3")] {
        let rows = csv.with("edge", edge);
        assert!(!rows.is_empty(), "no rows on edge {edge}");
        for r in rows {
            assert_eq!(
                csv.col(r, "stream"),
                stream,
                "edge {edge} carried stream {}",
                csv.col(r, "stream")
            );
        }
    }
    // A derived sample must never reach a consumer of the raw sweep, and a
    // sweep must never reach the derived consumer. Routing, both ways.
    for (stage, stream) in [("cloud", "1"), ("det-cloud", "3"), ("proc", "0")] {
        for r in csv.with("stage", stage) {
            assert_eq!(csv.col(r, "stream"), stream, "stage {stage}");
        }
    }

    // The far consumer did real work on the payload: it read points out of the
    // derived buffer and reported an address for it. A stage handed an
    // unreadable batch would report 0 points and storage_id 0 while every
    // count above still balanced.
    for r in csv.with("stage", "det-cloud") {
        assert_eq!(csv.col(r, "outcome"), "Delivered");
        assert_ne!(
            csv.col(r, "storage_id"),
            "0",
            "det-cloud could not find a point buffer in the derived batch"
        );
        for col in ["enqueued_ns", "dequeued_ns", "proc_start_ns", "proc_end_ns"] {
            assert!(
                !csv.col(r, col).is_empty(),
                "{col} empty on a det-cloud row"
            );
        }
    }

    // Provenance: every derived sample names the sweep it came from.
    // `Sample::parent` existed from the first sprint and was `None`
    // everywhere and read nowhere until this chain; a field nothing reads can
    // be wrong indefinitely.
    let r = reduce_block(&cwd, "handoff");
    assert_eq!(u64_at(&r, "parent_mismatch"), 0);
    assert_eq!(u64_at(&r, "down_delivered"), far.len() as u64);
    assert_line(
        &o,
        &format!(
            "det-cloud parent = lidar on {} of {} samples",
            far.len(),
            far.len()
        ),
    );
    // And the half of `parent` that says WHICH sweep. The line above is the
    // stream half, and on its own it is close to vacuous: with
    // `parent: Some((s.stream, s.seq.wrapping_add(7)))` — every derived cloud
    // naming a sweep it did not come from, several of which do not exist —
    // the whole workspace still passed and that line still read "8 of 8".
    // This one joins the seq the far end read against the sweep `reduce`
    // actually reduced, and it is the only assertion here that can tell a
    // traceable result from an untraceable one.
    assert_eq!(
        r["parent_seq_equal"],
        serde_json::json!(true),
        "a derived cloud named a sweep it was not built from"
    );
    assert_line(
        &o,
        &format!(
            "det-cloud parent seq = the sweep reduce read: OK ({} samples)",
            far.len()
        ),
    );
}

/// The derived stream is admitted on exactly the same terms as a sensor's:
/// through the same `Admission`, into the same dense `arrival_seq`, with its
/// own per-edge denominator and its own conservation line.
///
/// The `produced == admitted_of(LIDAR_DET)` half is the one that bites. A
/// chain whose results never reached admission would still let `reduce` report
/// everything it built, and every other number in the run would be unchanged —
/// it would simply be a stage talking to itself.
#[test]
fn the_derived_stream_is_admitted_like_a_sensors() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "admit", "--cap", "16"]);
    assert_ok(&o);
    let out = stdout(&o);
    assert!(!out.contains("-> FAIL"), "an invariant FAILed:\n{out}");

    // What `reduce` actually consumed and produced. `velo->reduce` is a fixed
    // cap-2 Drop* edge and these runs are unpaced, so `delivered == N` is a
    // statement about the scheduler rather than about the chain -- asserting it
    // made this test fail about one run in seven for the wrong reason. Every
    // line below is still exact, because each is computed from that
    // denominator rather than softened.
    let r = reduce_block(&cwd, "admit");
    let delivered = u64_at(&r, "delivered");
    let produced = u64_at(&r, "produced");
    assert!(produced > 0, "the chain produced nothing at all");
    // The third link admits too, so the global total counts four producers.
    // Read from the summary rather than assumed to be `produced` for the same
    // reason `produced` is not assumed to be `N`: `det->detect` is a fixed
    // cap-4 Drop* edge and may legitimately evict on an unpaced run.
    let detected = u64_at(&r["detect"], "produced");
    assert!(detected > 0, "the detection stage produced nothing at all");
    for line in [
        format!("INVARIANT stage=reduce delivered={delivered} produced={produced} errors=0 -> OK"),
        // FOUR producers now: two drivers and two consumer threads that
        // admit. The drivers admit N each whatever an edge later does with the
        // sample; each derived stream contributes exactly what it produced,
        // which is the half that bites -- a chain whose results never reached
        // admission leaves this total at 2N.
        format!(
            "INVARIANT admitted total={} admission={} -> OK",
            N as u64 * 2 + produced + detected,
            N as u64 * 2 + produced + detected
        ),
        // Conservation on the input edge against its own denominator: an
        // eviction moves a sweep from `delivered` to `dropped` and neither
        // leaves the edge.
        format!(
            "INVARIANT edge=velo->reduce delivered={delivered} dropped={} admitted={N} -> OK",
            N as u64 - delivered
        ),
        // Exact even if the edge evicted, because an eviction writes the
        // pusher's row.
        format!("INVARIANT rows edge=det->cloud rows={produced} admitted={produced} -> OK"),
    ] {
        assert!(
            out.lines().any(|l| l == line),
            "missing invariant:\n  {line}\n--- stdout ---\n{out}"
        );
    }

    // The derived edge's denominator is its own, not the global 24.
    let det_line = out
        .lines()
        .find(|l| l.starts_with("INVARIANT edge=det->cloud "))
        .unwrap_or_else(|| panic!("no det->cloud invariant:\n{out}"));
    assert!(
        det_line.ends_with(&format!("admitted={produced} -> OK")),
        "{det_line}"
    );

    // `arrival_seq` is one dense order over all FOUR streams, both derived
    // ones included. A second `Admission` for a derived stream — the cheap
    // alternative — restarts the counter and fails here.
    let csv = evidence(&cwd, "admit");
    let mut seqs: Vec<u64> = csv
        .rows
        .iter()
        .filter(|r| !csv.col(r, "edge").contains("->"))
        .filter(|r| !csv.col(r, "arrival_seq").is_empty())
        .map(|r| csv.col(r, "arrival_seq").parse().expect("arrival_seq"))
        .collect();
    seqs.sort_unstable();
    assert_eq!(
        seqs,
        (0..N as u64 * 2 + produced + detected).collect::<Vec<u64>>(),
        "arrival_seq is not one dense order over the four streams"
    );

    // And the derived samples are genuinely interleaved with the sensors'
    // rather than taking a block at the end — `reduce` admits as each sweep
    // arrives, not in a drain after everything else has finished.
    let mut order: Vec<(u64, String)> = csv
        .rows
        .iter()
        .filter(|r| !csv.col(r, "edge").contains("->"))
        .filter(|r| !csv.col(r, "arrival_seq").is_empty())
        .map(|r| {
            (
                csv.col(r, "arrival_seq").parse().expect("arrival_seq"),
                csv.col(r, "stream"),
            )
        })
        .collect();
    order.sort_by_key(|(s, _)| *s);
    let last_sensor = order
        .iter()
        .rposition(|(_, s)| s != "3")
        .expect("no sensor sample at all");
    let first_derived = order
        .iter()
        .position(|(_, s)| s == "3")
        .expect("no derived sample at all");
    assert!(
        first_derived < last_sensor,
        "every derived sample was admitted after every sensor sample, so the \
         chain ran as a drain rather than as a stage: {order:?}"
    );
}

/// A derived stream must outlive nothing and be outlived by nothing: its edge
/// is closed only after the stage that pushes to it has been joined.
///
/// **This is the failure that passes its own check.** Close `det->cloud`
/// beside the other queues — the obvious minimal edit — and every cloud
/// `reduce` admits during its drain comes back `PushOutcome::Closed`, which is
/// counted as a drop and written as a drop row with reason `closed`. So
/// `delivered + dropped == admitted` still balances, every invariant still
/// prints `-> OK`, and the derived samples have simply vanished from the
/// measurement.
///
/// **The fixture is the test.** My first version of this used the default
/// four-to-eleven-point sweeps, and I verified it against the bug: I moved
/// `det_q.close()` up beside the other queue closes and **the test still
/// passed, seven of seven green**. On sweeps that small `reduce` drains faster
/// than the driver can read files, so the queue is already empty when the
/// close lands, the race never happens, and the assertion below is about
/// nothing.
///
/// So the sweeps are large enough to invert that: `reduce` is the slowest
/// stage in the run by a wide margin, the driver finishes long before it does,
/// and a premature close is certain to catch samples in flight. Re-running the
/// same mutation against this version fails, which is the only reason to
/// believe the test at all.
#[test]
fn the_derived_edge_outlives_its_producer() {
    // 12 sweeps of 50k points against 4 frames: the driver's work per sweep is
    // one 800 KB read, the reduce stage's is 50k index-sort-accumulate in a
    // debug build. Orders of magnitude apart, in the direction that matters.
    let fx = FixtureDrive::with_velodyne_sized(4, 12, PERIOD_NS, |_| 50_000).unwrap();
    let cwd = TempDir::new().unwrap();
    // `--cap 1 --consumer-delay-ms 20` on the camera is the second control,
    // borrowed from `lidar.rs`: four frames through a cap-1 queue whose
    // consumer sleeps 20 ms each cannot all fit, so this run is GUARANTEED to
    // write drop rows carrying a reason. Without it, "no row says `closed`"
    // would pass against a binary that never fills the column at all.
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "shutdown",
            "--cap",
            "1",
            "--consumer-delay-ms",
            "20",
        ],
    );
    assert_ok(&o);
    let csv = evidence(&cwd, "shutdown");
    let r = reduce_block(&cwd, "shutdown");

    // Every derived cloud is accounted for on its edge, whatever became of it.
    // Not `== 12`: `velo->reduce` is a cap-2 Drop* edge and a stage this slow
    // will legitimately be outrun, so the denominator is what `reduce`
    // actually produced.
    let rows = csv.with("edge", "det->cloud");
    assert_eq!(
        rows.len() as u64,
        u64_at(&r, "produced"),
        "a derived cloud went unaccounted for on its own edge"
    );

    // The control fires first, so the assertion under it can never be vacuous.
    let drops = csv.with("stage", "admission");
    assert!(
        !drops.is_empty(),
        "nothing was dropped anywhere, so the `reason` column was never \
         exercised and the assertion below would be vacuous"
    );

    let closed: Vec<String> = csv
        .rows
        .iter()
        .filter(|r| csv.col(r, "reason") == "closed")
        .map(|r| format!("{} seq {}", csv.col(r, "edge"), csv.col(r, "seq")))
        .collect();
    assert!(
        closed.is_empty(),
        "{} sample(s) were pushed onto a queue that had already been closed \
         ({closed:?}) — a producer outlived its own edge, so those samples were \
         counted as drops and the run still reported OK",
        closed.len()
    );
    // Non-vacuous the other way too: the chain really ran.
    assert!(
        !csv.with("stage", "det-cloud").is_empty(),
        "no derived sample was delivered, so 'none was closed' says nothing"
    );
    assert!(u64_at(&r, "produced") > 0);
    assert!(!stdout(&o).contains("-> FAIL"));
}

/// Determinism, through the real binary: the same sweep reduced in two
/// separate runs produces the same-sized result, sweep for sweep.
///
/// The unit tests in `pipes_kitti::voxel` pin the output bytes; this pins that
/// nothing in the *pipeline* — thread interleaving, admission order, which
/// sweep a queue happened to evict — reaches the arithmetic.
///
/// **Two things were wrong with the first version of this test, and both are
/// worth writing down.** It built a map of `seq -> storage_id` and then threw
/// the value away (`.map(|(seq, _)| (seq, seq))`), so it mapped each sequence
/// number to itself; and it never compared the two maps to each other at all.
/// What it actually tested was eight aggregate `summary.json` fields, which
/// are means over each run's own population — so it also had to assert that
/// both runs consumed all `N` sweeps, which on an unpaced run over a cap-2
/// Drop\* edge is a statement about the scheduler rather than about the
/// transform.
///
/// The join key is `tov_start_ns`: the sweep's own time of validity,
/// inherited unchanged through the chain. A derived sample's `seq` is the
/// sweep's number too, so it would join the same way; the instant is the
/// measurement itself, and it cannot be renumbered by a stage.
///
/// What this pins is the *size* of each result — `bytes_alloc` on the producer
/// row is the output buffer — and not its contents: two runs that put the same
/// number of points in different cells would pass here. Value equality is
/// pinned in `pipes_kitti::voxel`'s unit tests, on the bytes themselves.
#[test]
fn two_runs_of_one_drive_reduce_to_the_same_result() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let built_by_tov = |name: &str| -> std::collections::BTreeMap<String, String> {
        assert_ok(&pipes(&cwd, &fx, &["--name", name, "--cap", "16"]));
        let csv = evidence(&cwd, name);
        csv.with("edge", "det")
            .iter()
            .filter(|r| csv.col(r, "stage") == "reduce")
            .map(|r| (csv.col(r, "tov_start_ns"), csv.col(r, "bytes_alloc")))
            .collect()
    };
    let a = built_by_tov("det-a");
    let b = built_by_tov("det-b");
    assert!(!a.is_empty() && !b.is_empty(), "a run produced nothing");

    let common: Vec<&String> = a.keys().filter(|k| b.contains_key(*k)).collect();
    assert!(
        common.len() >= 2,
        "the two runs share {} sweeps, so the comparison below would be about \
         nothing: {a:?} against {b:?}",
        common.len()
    );
    for k in common {
        assert_eq!(
            a[k], b[k],
            "the sweep at tov {k} reduced to a result of {} bytes in one run \
             and {} in the other",
            a[k], b[k]
        );
    }

    // Two population-independent facts the per-sweep comparison does not
    // cover: the parameter itself, and that neither run threw a point away —
    // a discarded point would make every ratio in the summary a filter's
    // rather than the grid's.
    let (ra, rb) = (reduce_block(&cwd, "det-a"), reduce_block(&cwd, "det-b"));
    assert_eq!(ra["voxel_size_m"], rb["voxel_size_m"]);
    for r in [&ra, &rb] {
        assert_eq!(u64_at(r, "non_finite_points"), 0);
        assert_eq!(u64_at(r, "out_of_range_points"), 0);
        assert_eq!(f64_at(r, "discarded_points_fraction"), 0.0);
    }
}

/// The parameter travels with the run and with the data, and an invalid one is
/// refused where it enters rather than producing an empty cloud in silence.
#[test]
fn the_voxel_size_is_recorded_and_validated() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "vox", "--cap", "16", "--voxel-size-m", "0.5"],
    );
    assert_ok(&o);
    // `run.json` says what was asked; `summary.json` says what it did and why
    // that number was chosen, so neither file needs the command line that made
    // it in order to be read.
    assert_eq!(
        run_json(&cwd, "vox")["voxel_size_m"],
        serde_json::json!(0.5)
    );
    let r = reduce_block(&cwd, "vox");
    assert_eq!(r["voxel_size_m"], serde_json::json!(0.5));
    assert!(
        r["voxel_size_rationale"]
            .as_str()
            .is_some_and(|s| s.contains("pedestrian")),
        "the summary does not say where the default came from: {r}"
    );
    // The rationale must not be glued to a size it does not justify. This run
    // asked for 0.5 m, and 0.5 is not one third of 0.6 — the line used to say
    // it was, because the rationale was a constant printed beside whatever
    // number the flag carried.
    assert_line(&o, "reduce voxel = 0.5 m (set with --voxel-size-m; the default 0.20 m is 1/3 of a 0.6 m pedestrian width, the smallest KITTI object class that must survive as a multi-voxel structure)");
    // The positive control: at the default the same line does claim it.
    let d = pipes(&cwd, &fx, &["--name", "voxdefault", "--cap", "16"]);
    assert_ok(&d);
    assert_line(&d, "reduce voxel = 0.2 m (the default 0.20 m is 1/3 of a 0.6 m pedestrian width, the smallest KITTI object class that must survive as a multi-voxel structure)");

    // Every rejected value fails SILENTLY further in — 0 puts every point out
    // of range, a negative edge runs the grid backwards, NaN discards
    // everything — so each must be refused at the boundary. clap exits 2 on a
    // bad argument, which is a different 2 from an invariant FAIL: the run
    // never starts, so there is no INVARIANT line at all.
    for bad in ["0", "-0.2", "NaN", "inf", "banana"] {
        let o = pipes(
            &cwd,
            &fx,
            &["--name", "badvox", "--cap", "16", "--voxel-size-m", bad],
        );
        assert!(
            !o.status.success(),
            "--voxel-size-m {bad} was accepted:\n{}",
            stdout(&o)
        );
        assert!(
            !stdout(&o).contains("INVARIANT"),
            "--voxel-size-m {bad} started a run before being refused"
        );
    }
}

/// A grid so fine that the reduction becomes a filter must say so **on the
/// shrink line**, not three lines below it.
///
/// `--voxel-size-m 1e-9` is finite and positive, so `parse_voxel_size` accepts
/// it — correctly, since nothing about the number itself is invalid. What it
/// addresses is +-1 mm, so on a real drive every point falls outside the
/// packed grid, the stage produces clouds of no points at all, and the run
/// printed
///
/// ```text
/// reduce points/sweep in=121794 out=0 (n/a)
/// byte chain shrink = 11883.34x smaller
/// ```
///
/// — the largest ratio this stage has ever reported, over an empty result,
/// with `-> OK` under it and exit 0. The discard count was printed, but three
/// lines away, and a reader who quotes the headline quotes a filter as a
/// reduction. The share discarded now travels with the ratio.
///
/// The negative control is in the same test and is the half that makes it
/// mean something: a normal run's shrink line must NOT carry the annotation,
/// or the annotation is just decoration that says nothing about this run.
#[test]
fn a_grid_that_discards_its_input_says_so_beside_the_ratio() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();

    // The control first. The fixture's own points are 1 m apart, nothing is
    // discarded at any sane grid, and the line stays clean.
    let good = pipes(&cwd, &fx, &["--name", "sane", "--cap", "16"]);
    assert_ok(&good);
    let good_line = line_with(&stdout(&good), "byte chain shrink = ").to_string();
    assert!(
        !good_line.contains("DISCARDED"),
        "a run that discarded nothing was annotated as though it had: {good_line}"
    );
    assert_eq!(
        f64_at(&reduce_block(&cwd, "sane"), "discarded_points_fraction"),
        0.0
    );

    // And now the degenerate one. The fixture's sweeps run out to ~11 m, so
    // at 1e-9 m every point but the one at the origin is outside the grid.
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "tiny", "--cap", "16", "--voxel-size-m", "1e-9"],
    );
    assert_ok(&o);
    let r = reduce_block(&cwd, "tiny");
    assert!(
        u64_at(&r, "out_of_range_points") > 0,
        "1e-9 m discarded nothing, so this test is about nothing: {r}"
    );
    // 59 of the fixture's 60 points: only the one at the origin is inside a
    // grid that addresses +-1 mm. (On a real drive it is all of them.)
    let discarded = f64_at(&r, "discarded_points_fraction");
    assert!(
        discarded > 0.9,
        "only {discarded} of the input was discarded"
    );
    let line = line_with(&stdout(&o), "byte chain shrink = ").to_string();
    assert!(
        line.contains("DISCARDED"),
        "a run that threw away all but a handful of its points reported its \
         shrink with no qualification: {line}"
    );
}
