//! Evidence rows (design §4): one row per sample per edge, written by the
//! stage that decided the sample's fate, plus the rare `Event`s. Host
//! timestamps are ns relative to `t0_host`; duration columns
//! (`queue_wait_ns`, `measurement_age_ns`, `decode_ns`, `push_blocked_ns`) are
//! elapsed ns. `queue_wait_ns` and
//! `measurement_age_ns` are materialised in the constructors here and
//! nowhere else. Field order is the CSV column order.

use std::sync::Arc;

use serde::Serialize;

use crate::clock::{HostTime, Tov};
use crate::queue::{PushOutcome, QueuePolicy};
use crate::sample::{payload_bytes, Sample, StreamId};

/// What happened to one sample on one edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Outcome {
    /// The sample reached the stage and was processed.
    Delivered,
    /// Evicted from a full DropOldest queue to make room for a newer sample.
    DroppedOldest,
    /// Refused by a full DropNewest queue, or by a full Block queue via a
    /// non-blocking `try_push`.
    DroppedNewest,
    /// A Block push gave up after `max_wait`.
    Timeout,
    /// The driver never produced the frame: its deadline had already passed,
    /// decoding failed, or the source has no sample for it at all
    /// (`absent_in_source`).
    Missing,
}

/// One evidence row. `Option` columns are empty in the CSV when not applicable.
#[derive(Clone, Debug, Serialize)]
pub struct Evidence {
    /// Run name, so rows from many runs can share one file.
    pub run_id: String,
    /// [`crate::clock::ClockModel`] epoch this sample was paced under.
    pub epoch: u32,
    /// Global admission order assigned under Admission's mutex, or `None` on a
    /// driver `Missing` row, which was never admitted.
    ///
    /// This was `u64` with 0 as the sentinel, which is wrong twice over: 0 is
    /// also the admission order of the *first* admitted sample, so a Missing
    /// row and the first Delivered row were indistinguishable in the column,
    /// and anyone counting rows by `arrival_seq` silently double-counted 0.
    /// `None` serialises to an empty CSV field, which cannot be confused with
    /// a real order.
    pub arrival_seq: Option<u64>,
    /// [`crate::sample::StreamId`] as a plain integer, for the CSV.
    pub stream: u8,
    /// Per-stream frame number from the driver.
    pub seq: u64,
    /// Which edge this row describes: `cam0` for the driver pseudo-edge,
    /// `cam0->proc` or `cam0->rerun` for a real queue.
    pub edge: &'static str,
    /// Which stage wrote the row: `driver`, `proc`, `rerun`, or `admission`
    /// for a drop the pusher recorded.
    pub stage: &'static str,
    /// Start of the sample's validity, ns since the Unix epoch on the SENSOR
    /// clock (0 when the sample has no time of validity).
    pub tov_start_ns: i64,
    /// End of validity; equal to `tov_start_ns` for an instant.
    pub tov_end_ns: i64,
    /// When the frame was scheduled, ns relative to `t0_host`; empty when
    /// unpaced (`rate_factor = inf`).
    pub due_ns: Option<i64>,
    /// When the driver finished producing the frame, ns relative to `t0_host`.
    /// `arrival_ns - due_ns` is the pacing error.
    pub arrival_ns: i64,
    /// When the sample was pushed onto this edge, ns relative to `t0_host`.
    pub enqueued_ns: Option<i64>,
    /// When the consumer popped it; empty on drop rows, which were never popped.
    pub dequeued_ns: Option<i64>,
    /// When the consumer began work on it.
    pub proc_start_ns: Option<i64>,
    /// When the consumer finished, including any artificial `--consumer-delay-ms`.
    pub proc_end_ns: Option<i64>,
    /// `dequeued - enqueued`; empty when dropped.
    pub queue_wait_ns: Option<i64>,
    /// `proc_end - due`; empty when dropped or unpaced.
    pub measurement_age_ns: Option<i64>,
    /// What happened to the sample on this edge.
    pub outcome: Outcome,
    /// Why, for non-`Delivered` outcomes: `evicted`, `full:drop-newest`,
    /// `full:block-nonblocking`, `max_wait`, `closed`, `deadline_skipped`,
    /// `decode_error`, `frame_error`, `absent_in_source` (a frame the
    /// sensor's source has no sample for; its `tov_*_ns` are 0, because the
    /// source measured nothing). Empty on `Delivered` rows.
    pub reason: &'static str,
    /// Queue length immediately BEFORE this push, so a drop row shows how full
    /// the queue was when the sample met it.
    pub depth_at_push: Option<u16>,
    /// Queue length immediately AFTER this sample was popped: the backlog the
    /// pop LEFT BEHIND, never counting the sample in this row. `Some(0)` means
    /// "this pop emptied the queue", not "one item was still there".
    ///
    /// The other edge of the occupancy cycle from
    /// [`Evidence::depth_at_push`], and the only one that can show a queue
    /// DRAINING. With the push depth alone, a queue that is permanently full
    /// and a queue that is full only for the instant of each push draw the
    /// same series, and a consumer catching up is invisible.
    ///
    /// `None` on every row whose sample never came out of a `pop`: driver
    /// rows, and every drop row. An eviction is not a drain -- the pusher's
    /// item takes the evictee's place and the depth is unchanged -- so a drop
    /// row stamping one would put a point on the drain series that no consumer
    /// ever produced.
    pub depth_after_pop: Option<u16>,
    /// Address of the frame's pixel buffer (`Buffer::as_ptr`), re-derived at
    /// each stage: equal across stages proves the pixels were shared, not
    /// copied. It is a per-frame identity check, NOT a unique id — the
    /// allocator reuses addresses once a frame is freed.
    pub storage_id: usize,
    /// Bytes requested from the allocator on this stage's own thread for this
    /// frame — not resident memory. Consumers measure their own output around
    /// the work (`Evidence::delivered`); driver rows are filled by `run.rs`
    /// from the `SLOT_DRIVER` delta between consecutive admits, which is one
    /// frame's PNG decode plus its Arrow batch build (plus anything else that
    /// thread did in that window — on the driver thread that is exactly the
    /// frame's work).
    ///
    /// A stage that both consumes and produces Arrow writes **two** rows per
    /// sample and splits this figure between them rather than repeating it:
    /// the consumer row (on the edge it read from) carries what receiving and
    /// reading the input cost, and the producer row (on the stream it emits)
    /// carries what building the output cost. Summing the column over a run
    /// therefore still totals each stage's allocations exactly once. The split
    /// is the point: it is what separates "the input was not copied" — the
    /// consumer row, which reads 0 for a borrow — from "nothing was
    /// allocated", which is false for any stage that genuinely transforms.
    pub bytes_alloc: u64,
    /// Arrow bytes this edge CARRIED for this sample, from
    /// [`crate::sample::payload_bytes`]: the size of the transfer itself.
    ///
    /// Taken from the sample on every row, so it reads the same at every stage
    /// that sees one buffer. That is the point rather than a redundancy: an
    /// unchanging carried size down a chain of stages IS the statement that
    /// nothing was copied.
    ///
    /// **Read it beside [`Evidence::bytes_alloc`], never instead of it.**
    /// `bytes_alloc` is what the stage REQUESTED FROM THE ALLOCATOR; this is
    /// what crossed the edge, and zero-copy is precisely the case where the
    /// two diverge -- 1,974,352 B carried against 0 B allocated. Either number
    /// on its own is compatible with the opposite conclusion: 0 allocated
    /// could mean nothing moved, and 1.97 MB carried could mean 1.97 MB was
    /// copied.
    ///
    /// 0 on a `Missing` row, which has no sample and so no transfer at all.
    pub payload_bytes: u64,
    /// PNG decode cost for this frame in ns, reported as a driver cost and
    /// never counted as pipeline latency.
    pub decode_ns: i64,
    /// Known only after a push returns: `Some` on the pusher's drop rows, empty on Delivered rows.
    pub push_blocked_ns: Option<i64>,
}

/// Per-run constants every row carries.
#[derive(Clone, Debug)]
pub struct RowCtx {
    /// Run name copied onto every row.
    pub run_id: String,
    /// Host-time origin every `_ns` column in a row is relative to.
    pub t0_host: HostTime,
    /// Clock epoch in force for the run.
    pub epoch: u32,
}

impl Evidence {
    fn rel(ctx: &RowCtx, h: HostTime) -> i64 {
        h - ctx.t0_host
    }

    /// The columns every row derives from the sample; queue/proc columns empty.
    fn base(ctx: &RowCtx, s: &Sample, edge: &'static str, stage: &'static str) -> Self {
        Evidence {
            run_id: ctx.run_id.clone(),
            epoch: s.epoch,
            arrival_seq: Some(s.arrival_seq),
            stream: s.stream.0,
            seq: s.seq,
            edge,
            stage,
            tov_start_ns: s.tov.start().map_or(0, |t| t.0),
            tov_end_ns: s.tov.end().map_or(0, |t| t.0),
            due_ns: s.due.map(|d| Self::rel(ctx, d)),
            arrival_ns: Self::rel(ctx, s.arrival),
            enqueued_ns: None,
            dequeued_ns: None,
            proc_start_ns: None,
            proc_end_ns: None,
            queue_wait_ns: None,
            measurement_age_ns: None,
            outcome: Outcome::Delivered,
            reason: "",
            depth_at_push: None,
            depth_after_pop: None,
            storage_id: s.storage_id,
            bytes_alloc: 0,
            // The size of the transfer, read off the sample every row is built
            // from. `payload_bytes` only sums buffer sizes the batch already
            // knows, so it allocates nothing and cannot perturb the
            // `bytes_alloc` window a caller may be inside.
            payload_bytes: payload_bytes(&s.payload) as u64,
            decode_ns: s.decode_ns,
            push_blocked_ns: None,
        }
    }

    /// Driver pseudo-edge row for a sample that was produced on time and
    /// admitted. `edge` is the producing stream's pseudo-edge (`cam0`,
    /// `velo`), and `stage` the driver that wrote it (`driver`, `velo-driver`).
    ///
    /// The stage string has to differ per stream, not only the edge: the
    /// `storage_id` cross-check in `run.rs` selects driver rows by `stage`, and
    /// two drivers both numbering from 0 under one stage string collide in a
    /// map keyed on `seq`.
    pub fn driver_admitted(
        ctx: &RowCtx,
        s: &Sample,
        edge: &'static str,
        stage: &'static str,
    ) -> Self {
        Self::base(ctx, s, edge, stage)
    }

    /// Driver pseudo-edge row for a sample that was never produced
    /// (`Missing`); `arrival` is when the driver made that decision.
    /// `storage_id` 0; no `arrival_seq`.
    ///
    /// `stream` used to be hard-coded to [`StreamId::CAM0`] with the comment
    /// "the only driver this sprint". A second driver made that a silent
    /// mislabel: every lidar sweep the velodyne driver skipped would have been
    /// filed in the evidence as a camera frame.
    #[allow(clippy::too_many_arguments)]
    pub fn driver_missing(
        ctx: &RowCtx,
        stream: StreamId,
        edge: &'static str,
        stage: &'static str,
        seq: u64,
        tov: Tov,
        due: Option<HostTime>,
        arrival: HostTime,
        reason: &'static str,
    ) -> Self {
        Evidence {
            run_id: ctx.run_id.clone(),
            epoch: ctx.epoch,
            arrival_seq: None,
            stream: stream.0,
            seq,
            edge,
            stage,
            tov_start_ns: tov.start().map_or(0, |t| t.0),
            tov_end_ns: tov.end().map_or(0, |t| t.0),
            due_ns: due.map(|d| Self::rel(ctx, d)),
            arrival_ns: Self::rel(ctx, arrival),
            enqueued_ns: None,
            dequeued_ns: None,
            proc_start_ns: None,
            proc_end_ns: None,
            queue_wait_ns: None,
            measurement_age_ns: None,
            outcome: Outcome::Missing,
            reason,
            depth_at_push: None,
            depth_after_pop: None,
            storage_id: 0,
            bytes_alloc: 0,
            // A frame that was never produced has no payload, so no edge
            // carried anything for it.
            payload_bytes: 0,
            decode_ns: 0,
            push_blocked_ns: None,
        }
    }

    /// Consumer row: popped, processed, done. Materialises `queue_wait_ns`
    /// and `measurement_age_ns`.
    ///
    /// Spec H.1's signature plus `depth_after_pop`, which is deliberately a
    /// parameter beside `depth_at_push` rather than a field the callers
    /// remember to assign afterwards. The two are one occupancy measurement
    /// read at its two ends, they arrive together on the same
    /// [`crate::queue::Queued`] envelope, and a caller that can pass one
    /// without being asked for the other is a caller that will ship the fill
    /// half of the series and not the drain half.
    #[allow(clippy::too_many_arguments)]
    pub fn delivered(
        ctx: &RowCtx,
        s: &Sample,
        edge: &'static str,
        stage: &'static str,
        enqueued: HostTime,
        depth_at_push: u16,
        depth_after_pop: Option<u16>,
        dequeued: HostTime,
        proc_start: HostTime,
        proc_end: HostTime,
        storage_id: usize,
        bytes_alloc: u64,
    ) -> Self {
        let mut e = Self::base(ctx, s, edge, stage);
        e.enqueued_ns = Some(Self::rel(ctx, enqueued));
        e.dequeued_ns = Some(Self::rel(ctx, dequeued));
        e.proc_start_ns = Some(Self::rel(ctx, proc_start));
        e.proc_end_ns = Some(Self::rel(ctx, proc_end));
        e.queue_wait_ns = Some(dequeued - enqueued);
        e.measurement_age_ns = s.due.map(|d| proc_end - d);
        e.depth_at_push = Some(depth_at_push);
        e.depth_after_pop = depth_after_pop;
        e.storage_id = storage_id;
        e.bytes_alloc = bytes_alloc;
        e
    }

    /// Pusher row for a sample that did not get through this edge. Signature per spec H.1.
    #[allow(clippy::too_many_arguments)]
    pub fn dropped(
        ctx: &RowCtx,
        s: &Sample,
        edge: &'static str,
        stage: &'static str,
        outcome: Outcome,
        reason: &'static str,
        enqueued: HostTime,
        depth_at_push: u16,
        push_blocked_ns: i64,
    ) -> Self {
        let mut e = Self::base(ctx, s, edge, stage);
        e.outcome = outcome;
        e.reason = reason;
        e.enqueued_ns = Some(Self::rel(ctx, enqueued));
        e.depth_at_push = Some(depth_at_push);
        e.push_blocked_ns = Some(push_blocked_ns);
        e
    }

    /// The spec G.3 mapping from a push outcome to the pusher's drop row.
    /// `Accepted` yields no row (the consumer writes the Delivered row);
    /// `Evicted` is a row for the EVICTED sample; the rest are rows for the
    /// new sample that was never enqueued.
    pub fn from_push(
        ctx: &RowCtx,
        edge: &'static str,
        stage: &'static str,
        policy: QueuePolicy,
        out: PushOutcome<Arc<Sample>>,
        push_blocked_ns: i64,
    ) -> Option<Self> {
        let (env, outcome, reason) = match out {
            PushOutcome::Accepted { .. } => return None,
            PushOutcome::Evicted(env) => (env, Outcome::DroppedOldest, "evicted"),
            PushOutcome::Rejected(env) => {
                let reason = if matches!(policy, QueuePolicy::Block { .. }) {
                    "full:block-nonblocking"
                } else {
                    "full:drop-newest"
                };
                (env, Outcome::DroppedNewest, reason)
            }
            PushOutcome::Timeout(env) => (env, Outcome::Timeout, "max_wait"),
            PushOutcome::Closed(env) => (env, Outcome::DroppedNewest, "closed"),
        };
        Some(Self::dropped(
            ctx,
            &env.item,
            edge,
            stage,
            outcome,
            reason,
            env.enqueued,
            env.depth_at_push,
            push_blocked_ns,
        ))
    }
}

/// Rare run-level events (D15); they go to `events.csv`, a separate schema.
#[derive(Clone, Copy, Debug, Serialize)]
pub enum EventKind {
    /// The evidence channel blocked for its full `max_wait` and switched to
    /// DropOldest; emitted at most once per run.
    RecorderDegraded,
    /// Reserved for an orderly shutdown notice (unused this sprint).
    Shutdown,
    /// A sensor's source has no sample for a run of consecutive frames (on
    /// KITTI's drive 0009, no lidar sweep for frames 177-180). Each such
    /// frame is its own `Missing` row with reason `absent_in_source`; this
    /// names the run once, when the replay reaches it, so the gap is read as
    /// one fact about the input rather than pieced together from its rows.
    SourceGap,
}

/// One row of `events.csv`.
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    /// Run name, matching the evidence rows.
    pub run_id: String,
    /// When the event happened, ns relative to `t0_host`.
    pub host_ns: i64,
    /// Admission order at the moment of the event, so it can be placed in the
    /// stream of samples.
    pub arrival_seq_after: u64,
    /// What happened.
    pub kind: EventKind,
    /// Which stage observed it.
    pub stage: &'static str,
    /// Human-readable detail for the CSV reader.
    pub detail: String,
}

impl Event {
    /// How a run ended, written as the last row of `events.csv` before the
    /// recorder is shut down.
    ///
    /// `detail` is `clean` or `panicked:<stage>`. There is deliberately no
    /// `invariant-failed` outcome: invariants are computed from the rows the
    /// recorder collected, which means they are only known *after* the sink is
    /// closed and `rec` has been joined — keeping the recorder alive until then
    /// would reopen the very window this event exists to close. The invariant
    /// result is in `summary.json` (`invariants_ok`) and the process exit code.
    pub fn shutdown(ctx: &RowCtx, host: HostTime, arrival_seq_after: u64, detail: &str) -> Self {
        Event {
            run_id: ctx.run_id.clone(),
            host_ns: host - ctx.t0_host,
            arrival_seq_after,
            kind: EventKind::Shutdown,
            stage: "run",
            detail: detail.to_string(),
        }
    }

    /// A run of frames the source has no sample for, reached by the replay at
    /// `host`: `detail` names the stream, the frames and the reason.
    pub fn source_gap(
        ctx: &RowCtx,
        host: HostTime,
        stage: &'static str,
        arrival_seq_after: u64,
        detail: String,
    ) -> Self {
        Event {
            run_id: ctx.run_id.clone(),
            host_ns: host - ctx.t0_host,
            arrival_seq_after,
            kind: EventKind::SourceGap,
            stage,
            detail,
        }
    }

    /// The one-per-run event written when `evidence->rec` degrades from
    /// `Block` to `DropOldest`.
    pub fn recorder_degraded(
        ctx: &RowCtx,
        host: HostTime,
        stage: &'static str,
        arrival_seq_after: u64,
    ) -> Self {
        Event {
            run_id: ctx.run_id.clone(),
            host_ns: host - ctx.t0_host,
            arrival_seq_after,
            kind: EventKind::RecorderDegraded,
            stage,
            detail: "evidence->rec push timed out; policy degraded to drop-oldest".to_string(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::datatypes::Schema;

    use super::*;
    use crate::clock::SensorTime;
    use crate::queue::BoundedQueue;

    fn ctx() -> RowCtx {
        RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        }
    }

    fn sample(seq: u64, storage_id: usize) -> Sample {
        Sample {
            stream: StreamId::CAM0,
            seq,
            arrival_seq: seq + 100,
            parent: None,
            tov: Tov::Time(SensorTime(1_000)),
            epoch: 0,
            due: Some(HostTime(500)),
            arrival: HostTime(510),
            payload: RecordBatch::new_empty(Arc::new(Schema::empty())),
            storage_id,
            decode_ns: 9,
        }
    }

    /// The 27 columns of `evidence.csv`, which is the sprint's raw record:
    /// every analysis script and every integration test reads it positionally
    /// or by name. Reordering a field of `Evidence` silently rewrites that
    /// file's schema, so the header is pinned here, in the crate that owns
    /// the struct, rather than only in the slow end-to-end test.
    #[test]
    fn evidence_csv_header_matches_the_struct() {
        let row = Evidence::driver_admitted(&ctx(), &sample(1, 7), "cam0", "driver");
        let mut w = csv::Writer::from_writer(Vec::new());
        w.serialize(&row).unwrap();
        let text = String::from_utf8(w.into_inner().unwrap()).unwrap();
        let header = text.lines().next().unwrap();
        assert_eq!(
            header,
            "run_id,epoch,arrival_seq,stream,seq,edge,stage,tov_start_ns,tov_end_ns,due_ns,arrival_ns,enqueued_ns,dequeued_ns,proc_start_ns,proc_end_ns,queue_wait_ns,measurement_age_ns,outcome,reason,depth_at_push,depth_after_pop,storage_id,bytes_alloc,payload_bytes,decode_ns,push_blocked_ns"
        );
        assert_eq!(header.split(',').count(), 26);
        // The data line has to line up with it, or the header could be right
        // while the row is not.
        let data = text.lines().nth(1).unwrap();
        assert_eq!(data.split(',').count(), 26);
        assert!(data.starts_with("t,0,101,0,1,cam0,driver,"));
        // The optional column at the end, `push_blocked_ns`, is empty rather
        // than 0 on a row that does not carry it, because 0 is a measurement:
        // "the push returned at once".
        assert!(data.ends_with(","), "push_blocked_ns: {data}");
        // A positive control for that assertion: a row that DOES carry it
        // must not end in an empty field, or the check above would pass
        // against a schema that had lost the column entirely.
        let mut filled = Evidence::driver_admitted(&ctx(), &sample(1, 7), "cam0", "driver");
        filled.push_blocked_ns = Some(1500);
        let mut w = csv::Writer::from_writer(Vec::new());
        w.serialize(&filled).unwrap();
        let text = String::from_utf8(w.into_inner().unwrap()).unwrap();
        let data = text.lines().nth(1).unwrap();
        assert!(data.ends_with(",1500"), "filled row: {data}");
    }

    #[test]
    fn delivered_materialises_wait_and_age() {
        let s = sample(3, 0xabc);
        let e = Evidence::delivered(
            &ctx(),
            &s,
            "cam0->proc",
            "proc",
            HostTime(520),
            2,
            Some(1),
            HostTime(600),
            HostTime(601),
            HostTime(700),
            0xabc,
            42,
        );
        assert_eq!(e.queue_wait_ns, Some(80));
        assert_eq!(e.measurement_age_ns, Some(200));
        assert_eq!(e.outcome, Outcome::Delivered);
        assert_eq!(e.depth_at_push, Some(2));
        // The other end of the same occupancy measurement: the backlog this pop
        // left behind. Carried as a parameter, so a caller cannot ship the fill
        // half of the series and silently omit the drain half.
        assert_eq!(e.depth_after_pop, Some(1));
        assert_eq!(e.push_blocked_ns, None);
        assert_eq!(e.bytes_alloc, 42);
        assert_eq!(
            (e.tov_start_ns, e.tov_end_ns, e.due_ns, e.arrival_ns),
            (1_000, 1_000, Some(500), 510)
        );
        assert_eq!(
            (e.arrival_seq, e.stream, e.seq, e.decode_ns),
            (Some(103), 0, 3, 9)
        );
    }

    #[test]
    fn driver_rows() {
        let s = sample(1, 7);
        let a = Evidence::driver_admitted(&ctx(), &s, "cam0", "driver");
        assert_eq!(
            (a.edge, a.stage, a.outcome, a.storage_id),
            ("cam0", "driver", Outcome::Delivered, 7)
        );
        assert_eq!(a.queue_wait_ns, None);
        let m = Evidence::driver_missing(
            &ctx(),
            StreamId::CAM0,
            "cam0",
            "driver",
            5,
            Tov::None,
            None,
            HostTime(9),
            "deadline_skipped",
        );
        assert_eq!(
            (m.seq, m.outcome, m.reason, m.storage_id, m.arrival_ns),
            (5, Outcome::Missing, "deadline_skipped", 0, 9)
        );
        assert_eq!((m.tov_start_ns, m.due_ns), (0, None));
        assert_eq!(
            (m.stream, m.edge, m.stage),
            (StreamId::CAM0.0, "cam0", "driver")
        );
        // The defect this signature exists to make impossible: `stream` was
        // hard-coded to cam0, so a second driver's skipped samples were filed
        // as camera frames. A cam0-only assertion above cannot see that, so a
        // second stream is asserted here too.
        let l = Evidence::driver_missing(
            &ctx(),
            StreamId::LIDAR,
            "lidar",
            "lidar-driver",
            2,
            Tov::Range {
                start: SensorTime(10),
                end: SensorTime(20),
            },
            None,
            HostTime(3),
            "deadline_skipped",
        );
        assert_eq!(
            (l.stream, l.edge, l.stage),
            (StreamId::LIDAR.0, "lidar", "lidar-driver")
        );
        assert_eq!((l.tov_start_ns, l.tov_end_ns), (10, 20));
    }

    #[test]
    fn from_push_mapping() {
        let c = ctx();
        let q = BoundedQueue::new(1, QueuePolicy::DropOldest);
        let s1 = Arc::new(sample(1, 11));
        let s2 = Arc::new(sample(2, 22));
        let out = q.try_push(Arc::clone(&s1), HostTime(100));
        assert!(Evidence::from_push(&c, "e", "admission", q.policy(), out, 5).is_none());
        let out = q.try_push(Arc::clone(&s2), HostTime(200));
        let row = Evidence::from_push(&c, "e", "admission", q.policy(), out, 5).unwrap();
        assert_eq!(
            (row.seq, row.outcome, row.reason),
            (1, Outcome::DroppedOldest, "evicted")
        );
        assert_eq!(
            (
                row.enqueued_ns,
                row.depth_at_push,
                row.push_blocked_ns,
                row.storage_id
            ),
            (Some(100), Some(0), Some(5), 11)
        );

        let qn = BoundedQueue::new(1, QueuePolicy::DropNewest);
        let _ = qn.try_push(Arc::clone(&s1), HostTime(1));
        let row = Evidence::from_push(
            &c,
            "e",
            "admission",
            qn.policy(),
            qn.try_push(Arc::clone(&s2), HostTime(2)),
            0,
        )
        .unwrap();
        assert_eq!(
            (row.seq, row.outcome, row.reason),
            (2, Outcome::DroppedNewest, "full:drop-newest")
        );

        let qb = BoundedQueue::new(
            1,
            QueuePolicy::Block {
                max_wait: std::time::Duration::from_millis(10),
            },
        );
        let _ = qb.try_push(Arc::clone(&s1), HostTime(1));
        let row = Evidence::from_push(
            &c,
            "e",
            "admission",
            qb.policy(),
            qb.try_push(Arc::clone(&s2), HostTime(2)),
            0,
        )
        .unwrap();
        assert_eq!(
            (row.outcome, row.reason),
            (Outcome::DroppedNewest, "full:block-nonblocking")
        );
        let row = Evidence::from_push(
            &c,
            "e",
            "admission",
            qb.policy(),
            qb.push(Arc::clone(&s2), HostTime(3)),
            7,
        )
        .unwrap();
        assert_eq!(
            (row.outcome, row.reason, row.push_blocked_ns),
            (Outcome::Timeout, "max_wait", Some(7))
        );

        qb.close();
        let row = Evidence::from_push(
            &c,
            "e",
            "admission",
            qb.policy(),
            qb.try_push(s2, HostTime(4)),
            0,
        )
        .unwrap();
        assert_eq!(
            (row.outcome, row.reason),
            (Outcome::DroppedNewest, "closed")
        );
    }

    #[test]
    fn recorder_degraded_event() {
        let ev = Event::recorder_degraded(&ctx(), HostTime(77), "proc", 9);
        assert!(matches!(ev.kind, EventKind::RecorderDegraded));
        assert_eq!(
            (ev.host_ns, ev.arrival_seq_after, ev.stage),
            (77, 9, "proc")
        );
    }
}
