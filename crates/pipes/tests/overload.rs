//! Overload behaviour of the `cam0->proc` edge, through the real binary.
//!
//! Forced by arithmetic, not by luck: an 8x4 PNG decodes in tens of
//! microseconds, so at `--rate inf` the driver offers all 8 frames in well
//! under 5 ms, while `--consumer-delay-ms 30` holds `proc` for 30 ms per
//! frame against a cap-1 queue. The margin between "driver outruns proc" and
//! "proc keeps up" is about three orders of magnitude.
//!
//! The *number* of drops still depends on the scheduler, so only bounds and
//! conservation identities are asserted here — never an exact drop count. The
//! exact, thread-free version of each policy lives in the unit tests in
//! `admission.rs` and `pipes-core::queue`.

use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{assert_ok, evidence, line_with, pipes, proc_stage, stdout, summary, u64_of, Csv};

/// Every `cam0->proc` row that is not the consumer's Delivered row.
fn drop_rows(csv: &Csv) -> Vec<&Vec<String>> {
    csv.rows
        .iter()
        .filter(|r| csv.col(r, "edge") == "cam0->proc" && csv.col(r, "outcome") != "Delivered")
        .collect()
}

#[test]
fn overload_drop_oldest_evicts_and_conserves() {
    let fx = FixtureDrive::new(8).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "dox",
            "--policy",
            "drop-oldest",
            "--cap",
            "1",
            "--consumer-delay-ms",
            "30",
        ],
    );
    // Exit success is the binary's own conservation check
    // (`delivered + dropped == admitted`, H.3).
    assert_ok(&o);

    let proc = proc_stage(&summary(&cwd, "dox"));
    let delivered = u64_of(&proc, "delivered");
    let oldest = u64_of(&proc, "dropped_oldest");
    assert!(
        oldest >= 1,
        "cap 1 against a 30 ms consumer dropped nothing: {proc}"
    );
    assert!(delivered >= 1, "nothing was delivered at all: {proc}");
    assert_eq!(delivered + oldest, 8, "{proc}");
    assert_eq!(u64_of(&proc, "dropped_newest"), 0, "{proc}");
    assert_eq!(u64_of(&proc, "timeout"), 0, "{proc}");

    let csv = evidence(&cwd, "dox");
    let drops = drop_rows(&csv);
    assert_eq!(drops.len() as u64, oldest, "summary and rows disagree");
    for r in drops {
        assert_eq!(csv.col(r, "outcome"), "DroppedOldest");
        assert_eq!(csv.col(r, "reason"), "evicted");
        // The evicted sample really was enqueued and really did meet a full
        // queue, so both columns must carry a measurement.
        assert!(
            !csv.col(r, "enqueued_ns").is_empty(),
            "seq {}",
            csv.col(r, "seq")
        );
        assert!(
            !csv.col(r, "depth_at_push").is_empty(),
            "seq {}",
            csv.col(r, "seq")
        );
    }
}

#[test]
fn overload_drop_newest_rejects_and_conserves() {
    let fx = FixtureDrive::new(8).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "dnw",
            "--policy",
            "drop-newest",
            "--cap",
            "1",
            "--consumer-delay-ms",
            "30",
        ],
    );
    assert_ok(&o);

    let proc = proc_stage(&summary(&cwd, "dnw"));
    let delivered = u64_of(&proc, "delivered");
    let newest = u64_of(&proc, "dropped_newest");
    assert!(
        newest >= 1,
        "cap 1 against a 30 ms consumer rejected nothing: {proc}"
    );
    assert_eq!(u64_of(&proc, "dropped_oldest"), 0, "{proc}");
    assert_eq!(u64_of(&proc, "timeout"), 0, "{proc}");
    assert_eq!(delivered + newest, 8, "{proc}");

    let csv = evidence(&cwd, "dnw");
    let drops = drop_rows(&csv);
    assert_eq!(drops.len() as u64, newest, "summary and rows disagree");
    for r in drops {
        assert_eq!(csv.col(r, "outcome"), "DroppedNewest");
        assert_eq!(csv.col(r, "reason"), "full:drop-newest");
    }
}

#[test]
fn overload_block_times_out_and_conserves() {
    let fx = FixtureDrive::new(8).unwrap();
    let cwd = TempDir::new().unwrap();
    // Forced by construction: cap 1, proc busy 60 ms per frame, max_wait
    // 20 ms => the third push must wait out its full 20 ms and give up, with
    // 3x margin.
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "blk",
            "--policy",
            "block",
            "--block-max-wait-ms",
            "20",
            "--cap",
            "1",
            "--consumer-delay-ms",
            "60",
        ],
    );
    assert_ok(&o);

    let s = summary(&cwd, "blk");
    let proc = proc_stage(&s);
    let delivered = u64_of(&proc, "delivered");
    let timeout = u64_of(&proc, "timeout");
    let newest = u64_of(&proc, "dropped_newest");
    assert!(
        timeout >= 1,
        "a blocking cap-1 edge never timed out: {proc}"
    );
    assert_eq!(u64_of(&proc, "dropped_oldest"), 0, "{proc}");
    assert_eq!(delivered + timeout + newest, 8, "{proc}");
    // Guaranteed at --rate inf: `due` is None, so the skip rule cannot fire
    // even while the driver is stalled on a blocking push.
    assert_eq!(s["missing"], serde_json::json!(0), "{s}");

    let csv = evidence(&cwd, "blk");
    let timeouts: Vec<_> = csv
        .rows
        .iter()
        .filter(|r| csv.col(r, "edge") == "cam0->proc" && csv.col(r, "outcome") == "Timeout")
        .collect();
    assert_eq!(timeouts.len() as u64, timeout, "summary and rows disagree");
    for r in &timeouts {
        assert_eq!(csv.col(r, "reason"), "max_wait");
        let blocked: i64 = csv.col(r, "push_blocked_ns").parse().unwrap();
        assert!(
            blocked >= 20_000_000,
            "a Timeout row that waited only {blocked} ns, under max_wait"
        );
    }

    // The bound is the whole point of Block being the negative control: the
    // driver really was stalled, and by about max_wait.
    let out = stdout(&o);
    let line = line_with(&out, "push_blocked_ns p99 = ");
    assert!(line.ends_with(" edge=cam0->proc"), "{line}");
    let p99: i64 = line
        .trim_start_matches("push_blocked_ns p99 = ")
        .trim_end_matches(" edge=cam0->proc")
        .parse()
        .unwrap_or_else(|e| panic!("{line}: {e}"));
    assert!(p99 >= 20_000_000, "push_blocked_ns p99 = {p99} ns: {line}");
}
