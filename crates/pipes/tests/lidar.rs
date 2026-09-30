//! Two producers through one admission, end to end, with no dataset.
//!
//! The claim this file exists to hold up is narrow and easy to fake: that the
//! lidar is a *genuine second producer* — concurrent with the camera, sharing
//! one admission mutex, on its own stream-routed edge with its own
//! denominators — rather than a second pipeline that happens to run in the
//! same process. Three things would let a fake pass, so each has a test that
//! bites:
//!
//! * A lidar replayed *after* the camera would still produce every row and
//!   every invariant. [`two_producers_interleave_under_one_admission`] pins
//!   the interleaving itself.
//! * A sweep pushed onto `cam0->proc` would be read as a camera frame.
//!   [`the_streams_do_not_cross_edges`] pins the routing.
//! * Every assertion here would pass trivially against a binary that never
//!   opens the lidar at all, so [`a_camera_only_drive_says_nothing_about_lidar`]
//!   is paired with the positive controls in the tests above it.
//!
//! `crates/pipes` is binary-only, so this drives the real binary through
//! `env!("CARGO_BIN_EXE_pipes")` with its CWD in a `TempDir`.
//!
//! **Nothing here asserts how many sweeps reached the consumer.** `--cap` is
//! the CAMERA's queue; `velo->cloud` is a fixed cap-2 DropOldest edge (see
//! `run::VELO_CAP` for why a ~1.95 MB payload does not get sixteen slots), so
//! under a loaded machine it legitimately evicts and a `dropped=0` assertion
//! would be asserting the scheduler. What holds unconditionally is
//! conservation — one evidence row per admitted sweep per edge, and
//! `delivered + dropped == admitted` — so that is what is pinned, and every
//! value assertion is computed from the sweeps that were *actually*
//! delivered, read back out of the evidence.

use pipes_kitti::testing::{fixture_sweep_points, FixtureDrive};
use tempfile::TempDir;

mod common;
use common::{assert_line, assert_ok, evidence, pipes, run_json, stderr, stdout, summary, Csv};

/// Frames and sweeps per fixture run, and the period they share.
///
/// 5 ms rather than the fixture's default 100 ms. Every test here but one is
/// unpaced, so the period is only the spacing of the instants in the files.
const N: usize = 8;
const PERIOD_NS: i64 = 5_000_000;

/// The period of the one PACED test, [`two_producers_interleave_under_one_admission`].
///
/// 25 ms, a 200 ms run. It was 5 ms, a 40 ms run, until a busy host started
/// the velodyne driver's thread 40 ms late: seven sweeps were past their
/// `skip_at` before the thread first ran, one was admitted, and the test read
/// a scheduler stall as the drivers being serialised. At 25 ms the same stall
/// would have to reach 175 ms to skip those seven sweeps.
const PACED_PERIOD_NS: i64 = 25_000_000;

fn lidar_fx() -> FixtureDrive {
    FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap()
}

fn camera_only_fx() -> FixtureDrive {
    FixtureDrive::with_period(N, PERIOD_NS).unwrap()
}

/// Every `INVARIANT …` line of a run, with its `-> OK|FAIL` suffix.
fn invariants(out: &str) -> Vec<String> {
    out.lines()
        .filter(|l| l.starts_with("INVARIANT "))
        .map(str::to_string)
        .collect()
}

/// `(stream, arrival_seq)` for every admitted sample, in admission order.
///
/// Selected by **producer pseudo-edge** — an edge name with no `->` in it —
/// rather than by a list of stage names. Consumer and admission rows carry an
/// `arrival_seq` too, so a stage list would have to be kept in step with every
/// stage that ever admits, and `reduce` writes rows on two edges of which only
/// one is a production. The structural test cannot fall out of date.
///
/// Driver `Missing` rows carry no `arrival_seq` and are excluded by the
/// emptiness test rather than by naming an outcome, so a new non-admitted
/// outcome cannot silently join the series.
fn admission_order(csv: &Csv) -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = csv
        .rows
        .iter()
        .filter(|r| !csv.col(r, "edge").contains("->"))
        .filter(|r| !csv.col(r, "arrival_seq").is_empty())
        .map(|r| {
            (
                csv.col(r, "stream"),
                csv.col(r, "arrival_seq").parse().expect("arrival_seq"),
            )
        })
        .collect();
    v.sort_by_key(|(_, seq)| *seq);
    v
}

/// How many times the stream changes along an admission order. `1` is two
/// contiguous blocks — one producer ran, then the other.
fn transitions(streams: &[&str]) -> usize {
    streams.windows(2).filter(|w| w[0] != w[1]).count()
}

/// The `seq`s the `cloud` consumer actually received, ascending.
///
/// Every value assertion about the lidar is computed over these rather than
/// over `0..N`: the edge may evict, and a test that assumed a full delivery
/// would fail on a busy machine for a reason that has nothing to do with what
/// it is checking.
fn delivered_sweeps(csv: &Csv) -> Vec<usize> {
    let mut v: Vec<usize> = csv
        .with("stage", "cloud")
        .iter()
        .map(|r| csv.col(r, "seq").parse().expect("seq"))
        .collect();
    v.sort_unstable();
    assert!(
        !v.is_empty(),
        "no sweep reached the cloud consumer at all — that is a broken edge, \
         not a busy machine"
    );
    v
}

/// The project's central claim for this step: one ordering decision
/// serialises two producers, so `arrival_seq` is a single dense order over
/// both streams and neither stream occupies a contiguous block of it.
///
/// **Paced, not `--rate inf`, and that is the point.** Unpaced, each driver
/// races through its whole drive as fast as the disk allows and one can
/// plausibly finish before the other starts — the run would look identical in
/// every other respect while the exercise was hollow. Paced, the two are on
/// the same schedule and genuinely contend.
///
/// The assertion is about *shape*, not about a particular order: which of two
/// samples due at the same instant wins the mutex is the scheduler's business
/// and asserting it would be asserting the scheduler. Its own fixture, at
/// [`PACED_PERIOD_NS`], so that a stall in that scheduler cannot pass for
/// one producer finishing before the other started.
#[test]
fn two_producers_interleave_under_one_admission() {
    let fx = FixtureDrive::with_velodyne(N, N, PACED_PERIOD_NS).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "duo", "--cap", "16", "--rate", "1"]);
    assert_ok(&o);

    let csv = evidence(&cwd, "duo");
    let order = admission_order(&csv);
    assert!(
        order.len() >= 4,
        "only {} samples were admitted at all:\n{}",
        order.len(),
        stdout(&o)
    );

    // Dense and unique: 0..n with nothing repeated and nothing skipped. A
    // second `Admission` per producer — the cheap alternative — restarts the
    // counter per stream and fails right here, with two rows at 0.
    let seqs: Vec<u64> = order.iter().map(|(_, s)| *s).collect();
    assert_eq!(
        seqs,
        (0..order.len() as u64).collect::<Vec<u64>>(),
        "arrival_seq is not one dense order over both streams"
    );

    // The interleaving claim is about the two SENSORS: two drivers on two
    // threads contending for one mutex. `reduce` admits a third stream into
    // the same order — which the density check above now covers, and which is
    // its own claim in `reduce.rs` — but it is a consumer, not a producer
    // racing the camera, so it is not part of this one.
    let streams: Vec<&str> = order
        .iter()
        .map(|(s, _)| s.as_str())
        .filter(|s| *s == "0" || *s == "1")
        .collect();
    let distinct: std::collections::BTreeSet<&str> = streams.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        2,
        "only one sensor stream admitted anything: {distinct:?}"
    );
    assert!(
        transitions(&streams) >= 3,
        "the streams did not interleave — {} transition(s) in {streams:?}; \
         one producer ran to completion before the other, so nothing was serialised",
        transitions(&streams)
    );
    // The negative control, on this run's own samples: sorted by stream they
    // are two contiguous blocks — exactly what a lidar replayed *after* the
    // camera would produce — and the predicate above rejects that. Without
    // this the threshold could be satisfied by any sequence at all and the
    // assertion would read as coverage it does not have.
    let mut blocked = streams.clone();
    blocked.sort_unstable();
    assert_eq!(
        transitions(&blocked),
        1,
        "the control is not two contiguous blocks: {blocked:?}"
    );

    // And the ranges overlap: each stream has a sample admitted after one of
    // the other's, both ways round. `transitions >= 3` implies this, but it
    // is the property in its own words, so a future weakening of the count
    // cannot quietly weaken the claim.
    let span = |name: &str| {
        let v: Vec<u64> = order
            .iter()
            .filter(|(s, _)| s == name)
            .map(|(_, q)| *q)
            .collect();
        (*v.first().expect(name), *v.last().expect(name))
    };
    let (cam_lo, cam_hi) = span("0");
    let (velo_lo, velo_hi) = span("1");
    assert!(
        cam_hi > velo_lo && velo_hi > cam_lo,
        "cam0 {cam_lo}..{cam_hi} and velo {velo_lo}..{velo_hi} do not overlap"
    );

    // The evidence carries two stream values, which is what makes every row
    // above attributable at all.
    assert!(!csv.with("stream", "0").is_empty());
    assert!(!csv.with("stream", "1").is_empty());
}

/// The negative control for `--reduce`: the M11 shape is still reachable, and
/// a run without the chain says nothing at all about it.
///
/// Without this, every assertion in `reduce.rs` would be about a code path
/// with no off switch, and nothing would show that the numbers this file pins
/// belong to the lidar rather than to the chain hanging off it.
#[test]
fn reduce_off_replays_the_lidar_with_leaf_consumers_only() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "leafonly", "--cap", "16", "--reduce", "off"],
    );
    assert_ok(&o);
    let out = stdout(&o);
    assert!(!out.contains("-> FAIL"), "an invariant FAILed:\n{out}");

    // The lidar is there and measured; the chain is not there at all.
    assert_line(&o, "velo admitted=8 missing=0 n_sweeps=8");
    assert_line(
        &o,
        &format!(
            "INVARIANT admitted total={} admission={} -> OK",
            N * 2,
            N * 2
        ),
    );
    for word in ["reduce ", "byte chain", "det->cloud", "det-cloud"] {
        assert!(
            !out.contains(word),
            "a run with --reduce off mentioned {word:?}:\n{out}"
        );
    }
    let csv = evidence(&cwd, "leafonly");
    for edge in ["velo->reduce", "det", "det->cloud"] {
        assert!(csv.with("edge", edge).is_empty(), "{edge} rows exist");
    }
    assert!(
        csv.with("stream", "3").is_empty(),
        "a derived sample exists"
    );
    assert!(
        summary(&cwd, "leafonly")["lidar"]["reduce"].is_null(),
        "summary.json claims a chain that did not run"
    );
    // And `run.json` records what was asked, so the directory is readable
    // without the command line that made it.
    assert_eq!(
        run_json(&cwd, "leafonly")["reduce"],
        serde_json::json!("off")
    );
}

/// A sweep must never reach `cam0->proc`, and a frame never `velo->cloud`.
///
/// Without the per-edge stream filter in `Admission::admit`, `proc` would be
/// handed a ~1.95 MB point cloud and read it as a camera frame — the failure is
/// not a crash but a plausible-looking wrong answer, which is the kind this
/// project exists to make impossible.
#[test]
fn the_streams_do_not_cross_edges() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "routed", "--cap", "16"]);
    assert_ok(&o);
    let csv = evidence(&cwd, "routed");

    for (edge, stream) in [
        ("cam0", "0"),
        ("cam0->proc", "0"),
        ("cam0->rerun", "0"),
        ("velo", "1"),
        ("velo->cloud", "1"),
    ] {
        let rows = csv.with("edge", edge);
        assert!(!rows.is_empty(), "no rows at all on edge {edge}");
        for r in rows {
            assert_eq!(
                csv.col(r, "stream"),
                stream,
                "edge {edge} carried stream {} (seq {})",
                csv.col(r, "stream"),
                csv.col(r, "seq")
            );
        }
    }
    // `proc` never wrote a row for anything but the camera, and `cloud` never
    // for anything but the lidar — the same claim from the stage's side, and
    // each stage must have written at least one row or the loop is vacuous.
    for (stage, stream) in [("proc", "0"), ("rerun", "0"), ("cloud", "1")] {
        let rows = csv.with("stage", stage);
        assert!(!rows.is_empty(), "stage {stage} wrote nothing");
        for r in rows {
            assert_eq!(csv.col(r, "stream"), stream, "stage {stage}");
        }
    }
}

/// The lidar's time of validity is an interval with the trigger inside it —
/// the case `Tov::Range` was written for and, until this step, never produced
/// by a running pipeline.
#[test]
fn velo_rows_carry_an_interval_and_the_camera_still_carries_an_instant() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    assert_ok(&pipes(&cwd, &fx, &["--name", "tov", "--cap", "16"]));
    let csv = evidence(&cwd, "tov");

    let velo = csv.with("stage", "velo-driver");
    assert_eq!(velo.len(), N, "one driver row per sweep");
    for r in &velo {
        let start: i64 = csv.col(r, "tov_start_ns").parse().unwrap();
        let end: i64 = csv.col(r, "tov_end_ns").parse().unwrap();
        assert_eq!(
            end - start,
            PERIOD_NS,
            "sweep {} does not span a rotation",
            csv.col(r, "seq")
        );
    }
    // The camera's rows are the control: an instant is a range of length 0,
    // so "start != end" is a real distinction rather than a property of every
    // row in the file.
    for r in csv.with("stage", "driver") {
        assert_eq!(
            csv.col(r, "tov_start_ns"),
            csv.col(r, "tov_end_ns"),
            "a camera frame grew an interval"
        );
    }

    // The third instant — the trigger — travels in the PAYLOAD, because
    // `Tov::Range` has room for two and the trigger is neither of them. The
    // consumer reads it back and reconciles it against the envelope it
    // arrived in, which is the only place in a run where those two records of
    // the same sweep's timing meet. The fixture puts the trigger at the
    // midpoint of the rotation; on drive_0005 it is 51.6 ms into 103.3 ms.
    let l = summary(&cwd, "tov")["lidar"].clone();
    assert_eq!(
        l["trigger_offset_mean_ns"].as_i64(),
        Some(PERIOD_NS / 2),
        "the trigger did not arrive where the fixture wrote it: {l}"
    );
    assert_eq!(l["trigger_outside_range"].as_u64(), Some(0));
    assert_eq!(
        l["trigger_missing"].as_u64(),
        Some(0),
        "a sweep arrived with no trigger, so the check above saw fewer \
         sweeps than it appears to"
    );
}

/// The per-edge denominators, read off the run's own output, plus the two
/// lines that tie the producers to admission.
#[test]
fn every_edge_is_measured_against_its_own_admitted_count() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "inv", "--cap", "16"]);
    assert_ok(&o);
    let out = stdout(&o);
    let inv = invariants(&out);

    assert!(!out.contains("-> FAIL"), "an invariant FAILed:\n{out}");
    // Each driver accounts for its own stream, and the two together account
    // for every `arrival_seq` handed out. `--cap 16` with 8 frames means
    // `cam0->proc` cannot drop, so that one line is exact; the lidar's edge
    // is cap 2 and its counts are not pinned (see the module note).
    // What the derived stream contributed to the global counter. Not `N`:
    // `velo->reduce` is a fixed cap-2 Drop* edge and these runs are unpaced,
    // so `reduce` being outrun is correct behaviour — it happened about one
    // run in seven and failed the total below for a reason that has nothing
    // to do with what this test checks. The two DRIVERS still admit `N` each
    // whatever an edge later does with the sample, so only this one term
    // moves.
    let reduce = summary(&cwd, "inv")["lidar"]["reduce"].clone();
    let produced = reduce["produced"]
        .as_u64()
        .unwrap_or_else(|| panic!("no lidar.reduce.produced in summary.json"));
    assert!(produced > 0, "the derived stream admitted nothing at all");
    // And the same for the third link, for the same reason: `detect` is also a
    // consumer thread that admits, so it also contributes to the one line
    // where the global counter is checked.
    let detected = reduce["detect"]["produced"]
        .as_u64()
        .unwrap_or_else(|| panic!("no lidar.reduce.detect.produced in summary.json"));
    assert!(detected > 0, "the detection stream admitted nothing at all");
    for line in [
        format!("INVARIANT driver admitted={N} missing=0 n_frames={N} -> OK"),
        format!("INVARIANT driver=velo admitted={N} missing=0 n_sweeps={N} -> OK"),
        // FOUR streams at the default: the camera, the lidar, the derived
        // cloud `reduce` admits and the detections `detect` admits. The total
        // is the one place the global counter is checked, and it has to count
        // every producer — including the two that are consumer threads.
        format!(
            "INVARIANT admitted total={} admission={} -> OK",
            N as u64 * 2 + produced + detected,
            N as u64 * 2 + produced + detected
        ),
        format!("INVARIANT edge=cam0->proc delivered={N} dropped=0 admitted={N} -> OK"),
        // One row per admitted sweep, whatever happened to it — exact even
        // when the edge evicted, because an eviction writes the pusher's row.
        format!("INVARIANT rows edge=velo->cloud rows={N} admitted={N} -> OK"),
    ] {
        assert!(inv.contains(&line), "missing:\n  {line}\nhave:\n{inv:#?}");
    }
    // The lidar edge's own denominator is 8, not the global 16 — and the
    // `-> OK` above already carries `delivered + dropped == 8`.
    let velo_line = inv
        .iter()
        .find(|l| l.starts_with("INVARIANT edge=velo->cloud "))
        .unwrap_or_else(|| panic!("no velo->cloud invariant:\n{inv:#?}"));
    assert!(
        velo_line.ends_with(&format!("admitted={N} -> OK")),
        "{velo_line}"
    );
    assert_line(
        &o,
        &format!("admitted = {}", N as u64 * 2 + produced + detected),
    );

    // The lidar's own zero-copy proof, and it is not the camera's. `NOT
    // CHECKED` here would mean the population was empty, which is the state
    // the camera's check would have silently reported as a pass.
    let storage = out
        .lines()
        .find(|l| l.starts_with("velo storage_id equal at 2 stages:"))
        .unwrap_or_else(|| panic!("no velo storage line:\n{out}"));
    assert!(
        storage.starts_with("velo storage_id equal at 2 stages: OK (")
            && !storage.contains("(0 sweeps)"),
        "{storage}"
    );
    assert_line(&o, "storage_id equal at 3 stages: OK");
    let edge_line = out
        .lines()
        .find(|l| l.starts_with("edge=velo->cloud "))
        .unwrap_or_else(|| panic!("no velo->cloud edge line:\n{out}"));
    assert!(
        edge_line.ends_with(&format!("admitted={N} storage_mismatch=0")),
        "{edge_line}"
    );
}

/// The number this whole step exists to make readable: what the edge CARRIES
/// against what the stage ALLOCATES, and which of the two scales with the
/// data.
///
/// Two runs, and the assertion is about the *difference*. A single expected
/// figure would have to encode arrow's own per-batch overhead — the metadata
/// columns, the offset pairs, the allocator's rounding — which is an
/// implementation detail of a dependency, so pinning it would make this fail
/// on an arrow upgrade while proving nothing about the pipeline. Solving for
/// that overhead in each run and requiring the two answers to agree cancels
/// all of it, and says the thing that matters: **carried bytes are the
/// points**.
///
/// The other half of the claim is in the same test: the consumer's
/// allocation does not move with the payload. Carried scales, allocated does
/// not — which is the zero-copy hand-off, and the baseline the next step has
/// to beat by making *carried* fall.
#[test]
fn carried_bytes_are_the_points_and_allocated_bytes_are_not() {
    let cwd = TempDir::new().unwrap();
    // A closure, so the two runs differ in exactly one thing: how many
    // sweeps, and therefore how many points crossed the edge.
    let run = |name: &str, n: usize| -> (u64, u64, u64) {
        let fx = FixtureDrive::with_velodyne(n, n, PERIOD_NS).unwrap();
        assert_ok(&pipes(&cwd, &fx, &["--name", name, "--cap", "32"]));
        let l = summary(&cwd, name)["lidar"].clone();
        let got = delivered_sweeps(&evidence(&cwd, name));
        // 16 B per point, over the sweeps that were actually delivered.
        let points: u64 = got.iter().map(|&i| fixture_sweep_points(i) as u64).sum();
        let carried = l["payload_bytes_total"].as_u64().expect("carried total");
        let allocated = l["cloud_bytes_per_sweep"].as_u64().expect("allocated");
        assert_eq!(
            l["delivered"].as_u64(),
            Some(got.len() as u64),
            "summary and evidence disagree on what was delivered"
        );
        assert!(
            carried > points * 16,
            "carried {carried} B does not even cover {points} points"
        );
        // Whatever is left once the points are accounted for is the fixed
        // per-batch header, and it must divide evenly into the sweeps.
        let overhead = carried - points * 16;
        assert_eq!(
            overhead % got.len() as u64,
            0,
            "the non-point bytes are not a per-sweep constant: {overhead} over {} sweeps",
            got.len()
        );
        (carried, overhead / got.len() as u64, allocated)
    };

    let (_, small_header, small_alloc) = run("bytes8", 8);
    let (_, large_header, large_alloc) = run("bytes24", 24);

    // Three times the sweeps, and the per-sweep cost of everything that is
    // not a point did not move. So carried bytes ARE the payload.
    assert_eq!(
        small_header, large_header,
        "the fixed part of a batch moved with the point count"
    );
    assert!(
        small_header < 4096,
        "per-sweep overhead is {small_header} B, which is not a fixed header"
    );

    // And the consumer allocated nothing, at either size. That 0 is only
    // meaningful beside the numbers above: a stage that allocated nothing
    // because it received nothing would report 0 here too — which is why
    // `delivered_sweeps` refuses an empty delivery.
    assert_eq!((small_alloc, large_alloc), (0, 0));
}

/// Each producer's allocations land on its own counter.
///
/// `SLOT_VELO_DRIVER` is a per-THREAD attribution, so a velodyne driver that
/// claimed the camera's slot would report 0 for itself and fold its sweeps
/// into the camera driver's `bytes_alloc` — in silence, and in the one column
/// the committed record is read for. That is the whole reason `N_SLOTS` grew
/// rather than the two drivers sharing one, and nothing else in this file
/// notices it: every other number stays exactly right.
#[test]
fn each_producer_allocates_on_its_own_counter() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    assert_ok(&pipes(&cwd, &fx, &["--name", "slots", "--cap", "16"]));
    let l = summary(&cwd, "slots")["lidar"].clone();
    let velo_driver = l["driver_bytes_per_sweep"].as_u64().expect("driver bytes");
    assert!(
        velo_driver > 0,
        "the velodyne driver reported {velo_driver} B/sweep — it reads a file \
         and builds an Arrow batch per sweep, so 0 means its thread is \
         counting against some other stage's slot"
    );

    // The camera's driver rows are the control: both producers allocate, and
    // each is reported under its own stage.
    let csv = evidence(&cwd, "slots");
    for stage in ["driver", "velo-driver"] {
        let total: u64 = csv
            .with("stage", stage)
            .iter()
            .map(|r| csv.col(r, "bytes_alloc").parse::<u64>().unwrap())
            .sum();
        assert!(total > 0, "stage {stage} allocated nothing at all");
    }
}

/// What the `cloud` consumer read out of the shared buffer, checked against
/// the fixture's closed form rather than against the implementation — and
/// over the sweeps it was actually handed, so an eviction cannot turn a
/// correct reading into a failure.
#[test]
fn the_consumer_reads_the_points_the_fixture_wrote() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    assert_ok(&pipes(&cwd, &fx, &["--name", "pts", "--cap", "16"]));
    let l = summary(&cwd, "pts")["lidar"].clone();
    let got = delivered_sweeps(&evidence(&cwd, "pts"));

    assert_eq!(l["storage_id_equal_all_stages"], serde_json::json!(true));
    assert_eq!(l["point_count_mismatch"].as_u64(), Some(0));

    let counts: Vec<usize> = got.iter().map(|&i| fixture_sweep_points(i)).collect();
    let lo = *counts.iter().min().unwrap();
    let hi = *counts.iter().max().unwrap();
    assert_eq!(l["points_min"].as_u64(), Some(lo as u64));
    assert_eq!(l["points_max"].as_u64(), Some(hi as u64));
    assert!(
        got.len() < 2 || lo != hi,
        "the fixture wrote {} sweeps of identical size, so min/max prove nothing",
        got.len()
    );

    // Point j of sweep i is (j, -j, i, 0.5), so the bounding box over the
    // delivered sweeps is x in [0, maxpts-1], y its negative, z the span of
    // the delivered indices.
    let x_hi = (hi - 1) as f64;
    let e = &l["extent"];
    assert_eq!(e["min"][0].as_f64(), Some(0.0));
    assert_eq!(e["max"][0].as_f64(), Some(x_hi));
    assert_eq!(e["min"][1].as_f64(), Some(-x_hi));
    assert_eq!(e["max"][1].as_f64(), Some(0.0));
    assert_eq!(e["min"][2].as_f64(), Some(got[0] as f64));
    assert_eq!(e["max"][2].as_f64(), Some(got[got.len() - 1] as f64));
}

/// Shutdown is producers first, then queues, then consumers — and a producer
/// is never outlived by its own edge.
///
/// **The fixture is deliberately lopsided: 8 frames against 48 sweeps.** The
/// camera's replay returns while the lidar still has forty files to read, so
/// at the instant the camera is done the second producer is certainly still
/// admitting. That is what makes this deterministic rather than a race the
/// test wins on a quiet machine: closing `velo->cloud` before joining its
/// driver produces `Closed` pushes every time, not sometimes.
///
/// And that failure is silent by construction. A `Closed` push increments the
/// queue's `dropped`, writes a drop row, and therefore keeps
/// `delivered + dropped == admitted` balanced and the row count exact — every
/// other assertion in this file still passes while the sweeps have vanished
/// from the measurement and the run reports `-> OK`. The `reason` column is
/// the only place it shows.
#[test]
fn a_producer_is_never_outlived_by_its_own_edge() {
    let fx = FixtureDrive::with_velodyne(8, 48, PERIOD_NS).unwrap();
    let cwd = TempDir::new().unwrap();
    // `--cap 1 --consumer-delay-ms 20` is the positive control, and it is on
    // the camera deliberately: 8 frames through a cap-1 queue whose consumer
    // sleeps 20 ms on each cannot all fit, so this run is GUARANTEED to write
    // drop rows carrying a reason. Without that, "no row says `closed`" would
    // pass just as well against a binary that never fills the column at all.
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

    // Every sweep is accounted for on both of its edges, whatever happened.
    assert_eq!(csv.with("edge", "velo").len(), 48);
    assert_eq!(csv.with("edge", "velo->cloud").len(), 48);

    // The control fires first, so a failure below can never be vacuous.
    let drops = csv.with("stage", "admission");
    assert!(
        !drops.is_empty(),
        "nothing was dropped anywhere, so the `reason` column was never \
         exercised and the assertion under this one would be vacuous"
    );
    for r in &drops {
        assert!(
            !csv.col(r, "reason").is_empty(),
            "a drop row with no reason: edge {} seq {}",
            csv.col(r, "edge"),
            csv.col(r, "seq")
        );
    }

    let closed: Vec<String> = csv
        .rows
        .iter()
        .filter(|r| csv.col(r, "reason") == "closed")
        .map(|r| format!("{} seq {}", csv.col(r, "edge"), csv.col(r, "seq")))
        .collect();
    assert!(
        closed.is_empty(),
        "{} sample(s) were pushed onto a queue that had already been closed \
         ({closed:?}) — a producer outlived its own edge, so those samples \
         were counted as drops while the run still reported OK",
        closed.len()
    );
}

/// A camera-only drive must run exactly as it did before this existed: no
/// flag, no warning, no empty lidar stream that reads as a real one with zero
/// sweeps.
///
/// This is the test that would pass against a binary with no lidar support at
/// all, which is why every other test in this file exists.
#[test]
fn a_camera_only_drive_says_nothing_about_lidar() {
    let fx = camera_only_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(&cwd, &fx, &["--name", "camonly", "--cap", "16"]);
    assert_ok(&o);

    let out = stdout(&o);
    assert!(
        !out.to_lowercase().contains("velo") && !out.to_lowercase().contains("lidar"),
        "a camera-only run mentioned the lidar:\n{out}"
    );
    assert!(
        stderr(&o).is_empty(),
        "a camera-only run wrote to stderr:\n{}",
        stderr(&o)
    );

    let s = summary(&cwd, "camonly");
    assert!(
        s["lidar"].is_null(),
        "a camera-only run claimed a lidar block: {}",
        s["lidar"]
    );
    let edges: Vec<&str> = s["stages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|st| st["edge"].as_str())
        .collect();
    assert_eq!(edges, vec!["cam0", "cam0->proc", "cam0->rerun"]);

    let csv = evidence(&cwd, "camonly");
    let streams: std::collections::BTreeSet<String> =
        csv.rows.iter().map(|r| csv.col(r, "stream")).collect();
    assert_eq!(
        streams,
        std::collections::BTreeSet::from(["0".to_string()]),
        "a camera-only run produced more than one stream"
    );

    // `run.json` still records what was ASKED, because `auto` means the answer
    // came off the disk and a reader needs to know which question was put.
    let j = run_json(&cwd, "camonly");
    assert_eq!(j["lidar"], serde_json::json!("auto"));
    assert!(j["n_sweeps"].is_null());
}

/// `--lidar` is an override, not a requirement, and it overrides in both
/// directions.
#[test]
fn the_lidar_flag_overrides_the_disk_both_ways() {
    // `off` on a drive that HAS lidar: the stream is ignored.
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &["--name", "off", "--cap", "16", "--lidar", "off"],
    );
    assert_ok(&o);
    assert!(
        !stdout(&o).contains("velo"),
        "--lidar off still replayed the lidar:\n{}",
        stdout(&o)
    );
    assert!(summary(&cwd, "off")["lidar"].is_null());
    assert_eq!(run_json(&cwd, "off")["lidar"], serde_json::json!("off"));

    // `on` on a drive that has NONE: an error naming the directory, not a
    // silent camera-only run.
    let bare = camera_only_fx();
    let cwd2 = TempDir::new().unwrap();
    let o2 = pipes(
        &cwd2,
        &bare,
        &["--name", "on", "--cap", "16", "--lidar", "on"],
    );
    assert!(
        !o2.status.success(),
        "--lidar on succeeded on a drive with no lidar:\n{}",
        stdout(&o2)
    );
    let err = stderr(&o2);
    assert!(
        err.contains("velodyne_points"),
        "the error does not name what is missing:\n{err}"
    );

    // And the control: the same `on` against a drive that does have one works.
    let cwd3 = TempDir::new().unwrap();
    let o3 = pipes(
        &cwd3,
        &lidar_fx(),
        &["--name", "on", "--cap", "16", "--lidar", "on"],
    );
    assert_ok(&o3);
    assert_eq!(run_json(&cwd3, "on")["lidar"], serde_json::json!("on"));
    assert_eq!(
        run_json(&cwd3, "on")["n_sweeps"],
        serde_json::json!(N as u64)
    );
}

/// The camera path is byte-identical with a second producer beside it.
///
/// `proc_bytes_per_frame` is the figure every one of the 39 committed rows
/// carries, and a run that changed it — by sharing a stage slot, by routing a
/// sweep through `proc`, by letting the lidar's back-pressure reach the camera
/// — would make those rows incomparable.
#[test]
fn the_camera_path_is_unchanged_by_the_lidar() {
    let cwd = TempDir::new().unwrap();
    let with = pipes(
        &cwd,
        &lidar_fx(),
        &["--name", "with", "--cap", "16", "--lidar", "on"],
    );
    assert_ok(&with);
    let without = pipes(
        &cwd,
        &camera_only_fx(),
        &["--name", "without", "--cap", "16"],
    );
    assert_ok(&without);

    for o in [&with, &without] {
        // 8x4 grayscale: one 32-byte output buffer per frame, whatever else
        // the process is doing.
        assert_line(o, "proc bytes/frame = 32");
        assert_line(
            o,
            &format!("edge=cam0->proc delivered={N} dropped=0 admitted={N} storage_mismatch=0"),
        );
        assert_line(o, "storage_id equal at 3 stages: OK");
    }
    let a = summary(&cwd, "with");
    let b = summary(&cwd, "without");
    for key in ["n_frames", "admitted", "missing", "policy"] {
        assert_eq!(a[key], b[key], "{key} moved when the lidar was added");
    }
    let proc_stage = |s: &serde_json::Value| {
        s["stages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|st| st["edge"] == serde_json::json!("cam0->proc"))
            .unwrap()
            .clone()
    };
    assert_eq!(
        proc_stage(&a)["bytes_alloc_per_frame_mean"],
        proc_stage(&b)["bytes_alloc_per_frame_mean"]
    );
    assert_eq!(proc_stage(&a)["delivered"], proc_stage(&b)["delivered"]);
}

/// The lidar's rows reach the evidence file whole: one per sweep on the
/// driver pseudo-edge and one per sweep on the consumer edge, with the
/// conservation identity holding on each.
#[test]
fn the_evidence_accounts_for_every_sweep_on_every_edge() {
    let fx = lidar_fx();
    let cwd = TempDir::new().unwrap();
    assert_ok(&pipes(&cwd, &fx, &["--name", "acct", "--cap", "16"]));
    let csv = evidence(&cwd, "acct");

    // Eleven edges at the default. Every admitted sample gets exactly one row
    // per edge it travels, whatever became of it — but "admitted" is not `N`
    // on every edge, and that is the distinction this test has to keep.
    //
    // Six edges carry a stream whose producer admits `N` samples: the three
    // camera edges and the three lidar ones (the driver pseudo-edge and its
    // two consumers). The next three carry the DERIVED CLOUD, whose producer
    // admits one per sweep that reached `reduce` — `N` only while
    // `velo->reduce`, a fixed cap-2 Drop* edge, evicts nothing, which on an
    // unpaced run it need not (the header says why). The last two carry the
    // DETECTIONS, one per cloud that reached `detect` across `det->detect`,
    // a fixed cap-4 Drop* edge on the same terms. Both derived counts are
    // read from the summary rather than assumed. This test once put the
    // cloud's three edges in the first group and failed on a loaded host
    // with 78 rows against 84: two sweeps evicted, six rows short.
    let n_edges = [
        "cam0",
        "cam0->proc",
        "cam0->rerun",
        "velo",
        "velo->cloud",
        "velo->reduce",
    ];
    let s = summary(&cwd, "acct");
    let produced_by = |stage: &serde_json::Value, what: &str| -> usize {
        let n = stage["produced"]
            .as_u64()
            .unwrap_or_else(|| panic!("no {what} produced count in summary.json: {s}"));
        assert!(n > 0, "the {what} stream admitted nothing at all");
        n as usize
    };
    let reduced = produced_by(&s["lidar"]["reduce"], "derived cloud");
    let detected = produced_by(&s["lidar"]["reduce"]["detect"], "detection");
    let det_edges = ["det", "det->cloud", "det->detect"];
    let obj_edges = ["obj", "obj->sink"];
    assert_eq!(
        csv.rows.len(),
        N * n_edges.len() + reduced * det_edges.len() + detected * obj_edges.len(),
        "rows against {N} sweeps, {reduced} clouds and {detected} detection batches"
    );
    for edge in n_edges {
        assert_eq!(csv.with("edge", edge).len(), N, "{edge} rows");
    }
    for edge in det_edges {
        assert_eq!(csv.with("edge", edge).len(), reduced, "{edge} rows");
    }
    for edge in obj_edges {
        assert_eq!(csv.with("edge", edge).len(), detected, "{edge} rows");
    }
    // Every row on the lidar's consumer edge is either the consumer's or the
    // pusher's; nothing comes from nowhere.
    let rows = csv.with("edge", "velo->cloud");
    let delivered = rows
        .iter()
        .filter(|r| csv.col(r, "stage") == "cloud")
        .count();
    let dropped = rows
        .iter()
        .filter(|r| csv.col(r, "stage") == "admission")
        .count();
    // Conservation, not a count. `--cap 16` is the CAMERA's queue; this edge
    // is a fixed cap-2 DropOldest one and may legitimately evict, so what is
    // pinned is that every admitted sweep is accounted for exactly once.
    assert_eq!(delivered + dropped, N, "a velo->cloud row from nowhere");
    assert!(delivered >= 1, "nothing reached the cloud consumer");

    // Every non-delivered row says why. The `closed` case -- a producer
    // outliving its own edge -- has its own test, where the fixture makes it
    // deterministic.
    for r in &csv.rows {
        if csv.col(r, "outcome") != "Delivered" {
            assert!(
                !csv.col(r, "reason").is_empty(),
                "a non-delivered row with no reason: edge {} seq {}",
                csv.col(r, "edge"),
                csv.col(r, "seq")
            );
        }
    }

    // A delivered sweep carries the columns a delivered sample must.
    for r in csv.with("stage", "cloud") {
        for col in ["enqueued_ns", "dequeued_ns", "proc_start_ns", "proc_end_ns"] {
            assert!(!csv.col(r, col).is_empty(), "{col} empty on a cloud row");
        }
        assert_eq!(csv.col(r, "outcome"), "Delivered");
        assert_ne!(csv.col(r, "storage_id"), "0");
    }
}
