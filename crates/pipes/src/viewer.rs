//! The viewer as a consumer: every picture the run draws -- each stage's and
//! the recorder's dashboard -- is a message on one bounded queue,
//! `draw->viewer`, and one thread, `viewer`, is the only code that calls the
//! Rerun SDK's `log`.
//!
//! **Why.** The SDK does not drop. Once about six seconds of drive 0005 are
//! buffered it blocks whoever called `log` until the viewer takes more, and
//! rerun 0.38.1 has no setting that makes it drop instead. When the stages
//! called `log` themselves, a viewer frozen for 10 s blocked them, their input
//! queues evicted, and the run answered 121 of 154 sweeps. Now a stage's `log`
//! only queues, the queue never waits, and a slow viewer loses drawings on its
//! own edge, recorded like any other consumer's losses.
//!
//! **How a stage draws.** A [`Canvas`] takes the calls a `RecordingStream`
//! takes -- the three timelines, `log`, `log_static` -- so the drawing code
//! reads as it did. `log` serializes the archetype into the component batches
//! the SDK would have built from it (the same call, `as_serialized_batches`,
//! so a picture that shares an Arrow buffer, like the camera frame, shares it
//! into the queue rather than copying it) and keeps them with the canvas's
//! timelines. [`Canvas::send`] pushes everything drawn since the last send as
//! one [`Drawing`]: one stage's picture of one sample. A drawing is taken whole
//! or dropped whole, so a `Clear` is never shown without the boxes that
//! replace it.
//!
//! **The record.** Each drawing is one evidence row on `draw->viewer`:
//! `Delivered` once the SDK has taken it, or `DroppedOldest` with reason
//! `evicted` when a newer drawing pushed it out of a full queue. One row per
//! drawing, not per `log` call and not a total per sample, so the edge
//! balances the way every other edge does -- `delivered + dropped ==
//! admitted`, the queue's own drop counter against the rows, one row per
//! admitted drawing -- and a row says whose picture of which sample was lost.
//! A total per sample would need a count column `evidence.csv` does not have,
//! and could only be written at the end of the run, since up to nine
//! producers draw a sample at different times. The viewer thread writes both
//! kinds of row: a push that evicts leaves the evicted drawing's particulars
//! behind for it. That keeps the recorder, which draws too, from writing rows
//! about a queue it pushes to, and the dashboard does not draw this edge's
//! rows at all, or each would be drawn, and each drawing would be a row.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use arrow::array::Array;
use pipes_core::alloc::{bytes_alloc, set_stage_slot, SLOT_VIEWER};
use pipes_core::clock::{now, HostTime};
use pipes_core::evidence::{Evidence, Outcome, RowCtx};
use pipes_core::queue::{BoundedQueue, PushOutcome, QueuePolicy, Queued};
use pipes_core::sample::Sample;
use rerun::log::{ChunkBatcherConfig, LogMsg};
use rerun::sink::{FileSink, LogSink, SinkFlushError};
use rerun::{
    AsComponents, EntityPath, RecordingStream, RecordingStreamBuilder, RecordingStreamResult,
    SerializedComponentBatch,
};

use crate::record::{EvRow, EvidenceSink};
use crate::run::RunCtx;

/// The viewer's edge: from every stage that draws to the one thread that
/// hands the drawings to the SDK.
pub const VIEWER_EDGE: &str = "draw->viewer";

/// The consumer of [`VIEWER_EDGE`], on its delivered rows.
pub const VIEWER_STAGE: &str = "viewer";

/// Capacity of [`VIEWER_EDGE`], in drawings: about 1.2 s of drive 0005's.
///
/// A default run of drive 0005 draws 3,391 drawings in 16 s, about 210 a
/// second and 22 MB a second with the camera's frames, and a viewer that
/// keeps up leaves at most 4 of them waiting (3 to 4 on the runs of
/// 2026-09-29, uncertified). The queue is not what rides out a stall: the SDK
/// holds about five seconds before `log` blocks (5.4 s of both 10 s freezes).
/// This depth buys two things. A push never meets a full queue in a healthy
/// run -- 256 is 60 times the deepest backlog seen -- and a stall holds a
/// bounded amount: the worst 256 drawings in a row were 29 MB, the camera's
/// frames among them held by reference, not copied. Deeper would keep older
/// pictures of a scene the viewer is already seconds behind, at more memory.
pub const VIEWER_CAP: usize = 256;

/// The policy of [`VIEWER_EDGE`]: a push never waits, and a full queue loses
/// its oldest drawing, so what the viewer shows after a stall is the present.
pub const VIEWER_POLICY: QueuePolicy = QueuePolicy::DropOldest;

/// One timeline's value, as the stage set it. Kept as the call the stage made
/// rather than as a finished `TimeCell`, so the viewer thread makes the same
/// call and the SDK converts it exactly as it did when the stage made it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Time {
    TimestampNs(i64),
    DurationSecs(f64),
    Sequence(i64),
}

/// How many timelines a canvas holds at once: the run draws on three,
/// `sensor_time`, `host` and `seq`.
const TIMELINES: usize = 3;

/// The timelines in force on a canvas, which a `RecordingStream` keeps per
/// thread and a canvas keeps per stage.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Stamps([Option<(&'static str, Time)>; TIMELINES]);

impl Stamps {
    fn set(&mut self, name: &'static str, t: Time) {
        let at = self
            .0
            .iter()
            .position(|s| s.is_some_and(|(n, _)| n == name))
            .or_else(|| self.0.iter().position(Option::is_none));
        debug_assert!(at.is_some(), "a canvas holds {TIMELINES} timelines");
        if let Some(i) = at {
            self.0[i] = Some((name, t));
        }
    }

    fn unset(&mut self, name: &'static str) {
        for s in &mut self.0 {
            if s.is_some_and(|(n, _)| n == name) {
                *s = None;
            }
        }
    }

    /// Makes these, and only these, the calling thread's timelines on `rec`.
    fn apply(&self, rec: &RecordingStream) {
        rec.reset_time();
        for &(name, t) in self.0.iter().flatten() {
            match t {
                Time::TimestampNs(ns) => rec.set_timestamp_nanos_since_epoch(name, ns),
                Time::DurationSecs(s) => rec.set_duration_secs(name, s),
                Time::Sequence(n) => rec.set_time_sequence(name, n),
            }
        }
    }
}

/// One `log` call, serialized: everything the SDK needs to take it.
struct Entry {
    path: EntityPath,
    /// `None` for `log_static`.
    time: Option<Stamps>,
    batches: Vec<SerializedComponentBatch>,
}

/// Which sample a drawing shows: the identity columns of its evidence row,
/// relative to the run's clock like every row's. Taken from the sample a
/// stage drew, or from the evidence row the dashboard drew.
#[derive(Clone, Debug, Default)]
pub struct Subject {
    epoch: u32,
    arrival_seq: Option<u64>,
    stream: u8,
    seq: u64,
    tov_start_ns: i64,
    tov_end_ns: i64,
    due_ns: Option<i64>,
    arrival_ns: i64,
    storage_id: usize,
    decode_ns: i64,
}

impl Subject {
    /// The sample a stage just drew.
    pub fn of_sample(rc: &RowCtx, s: &Sample) -> Self {
        Subject {
            epoch: s.epoch,
            arrival_seq: Some(s.arrival_seq),
            stream: s.stream.0,
            seq: s.seq,
            tov_start_ns: s.tov.start().map_or(0, |t| t.0),
            tov_end_ns: s.tov.end().map_or(0, |t| t.0),
            due_ns: s.due.map(|d| d - rc.t0_host),
            arrival_ns: s.arrival - rc.t0_host,
            storage_id: s.storage_id,
            decode_ns: s.decode_ns,
        }
    }

    /// The sample an evidence row is about, for the dashboard's drawing of it.
    pub fn of_row(e: &Evidence) -> Self {
        Subject {
            epoch: e.epoch,
            arrival_seq: e.arrival_seq,
            stream: e.stream,
            seq: e.seq,
            tov_start_ns: e.tov_start_ns,
            tov_end_ns: e.tov_end_ns,
            due_ns: e.due_ns,
            arrival_ns: e.arrival_ns,
            storage_id: e.storage_id,
            decode_ns: e.decode_ns,
        }
    }

    /// This drawing's row on [`VIEWER_EDGE`], queue and processing columns empty.
    fn row(&self, rc: &RowCtx, stage: &'static str, payload_bytes: u64) -> Evidence {
        Evidence {
            run_id: rc.run_id.clone(),
            epoch: self.epoch,
            arrival_seq: self.arrival_seq,
            stream: self.stream,
            seq: self.seq,
            edge: VIEWER_EDGE,
            stage,
            tov_start_ns: self.tov_start_ns,
            tov_end_ns: self.tov_end_ns,
            due_ns: self.due_ns,
            arrival_ns: self.arrival_ns,
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
            storage_id: self.storage_id,
            bytes_alloc: 0,
            payload_bytes,
            decode_ns: self.decode_ns,
            push_blocked_ns: None,
        }
    }
}

/// One stage's picture of one sample: every `log` it made since its last
/// [`Canvas::send`].
pub struct Drawing {
    subject: Subject,
    /// The stage that drew it.
    stage: &'static str,
    /// Arrow bytes its batches hold, a shared buffer counted whole: what the
    /// edge carries for it, as `payload_bytes` is on every other edge.
    bytes: u64,
    entries: Vec<Entry>,
}

/// A drawing that never reached the viewer, as the push that lost it left it
/// for the viewer thread to write up. The picture itself is freed at the push.
struct Lost {
    subject: Subject,
    stage: &'static str,
    bytes: u64,
    enqueued: HostTime,
    depth_at_push: u16,
    push_blocked_ns: i64,
    outcome: Outcome,
    reason: &'static str,
}

/// [`VIEWER_EDGE`] and its counters, shared by every canvas and the viewer
/// thread.
pub struct ViewerQueue {
    q: BoundedQueue<Drawing>,
    /// Every push, the denominator of `delivered + dropped == admitted`.
    admitted: AtomicU64,
    /// How long each push took, ns: never a wait, only the lock.
    push_ns: Mutex<Vec<i64>>,
    /// Drawings lost since the viewer thread last wrote their rows.
    lost: Mutex<Vec<Lost>>,
}

/// D9 poison policy, as everywhere: a panicked holder poisons nothing.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl ViewerQueue {
    pub fn new(cap: usize) -> Arc<Self> {
        Arc::new(ViewerQueue {
            q: BoundedQueue::new(cap, VIEWER_POLICY),
            admitted: AtomicU64::new(0),
            push_ns: Mutex::new(Vec::new()),
            lost: Mutex::new(Vec::new()),
        })
    }

    /// A canvas for `stage` to draw on.
    pub fn canvas(self: &Arc<Self>, stage: &'static str) -> Canvas {
        Canvas {
            viewer: Arc::clone(self),
            stage,
            time: Cell::new(Stamps::default()),
            pending: RefCell::new(Vec::new()),
        }
    }

    /// Never waits: a full queue evicts its oldest drawing, and a closed one
    /// refuses this one. Either way the lost drawing's picture is freed here
    /// and its row is left for the viewer thread.
    fn push(&self, d: Drawing) {
        let t = now();
        let out = self.q.try_push(d, t);
        let push_blocked_ns = now() - t;
        self.admitted.fetch_add(1, Relaxed);
        lock(&self.push_ns).push(push_blocked_ns);
        let (env, outcome, reason) = match out {
            PushOutcome::Accepted { .. } => return,
            PushOutcome::Evicted(env) => (env, Outcome::DroppedOldest, "evicted"),
            PushOutcome::Closed(env) => (env, Outcome::DroppedNewest, "closed"),
            // Neither happens under DropOldest; named as `Evidence::from_push` names them.
            PushOutcome::Rejected(env) => (env, Outcome::DroppedNewest, "full:drop-newest"),
            PushOutcome::Timeout(env) => (env, Outcome::Timeout, "max_wait"),
        };
        let Queued {
            item,
            enqueued,
            depth_at_push,
            ..
        } = env;
        let Drawing {
            subject,
            stage,
            bytes,
            entries,
        } = item;
        drop(entries);
        lock(&self.lost).push(Lost {
            subject,
            stage,
            bytes,
            enqueued,
            depth_at_push,
            push_blocked_ns,
            outcome,
            reason,
        });
    }

    /// No drawing is taken after this; the viewer thread drains what is
    /// queued and stops. Idempotent.
    pub fn close(&self) {
        self.q.close();
    }

    /// Drawings pushed.
    pub fn admitted(&self) -> u64 {
        self.admitted.load(Relaxed)
    }

    /// `BoundedQueue::dropped()`: every drawing a push did not deliver.
    pub fn dropped(&self) -> u64 {
        self.q.dropped()
    }

    /// Every push's duration, ns.
    pub fn push_ns(&self) -> Vec<i64> {
        lock(&self.push_ns).clone()
    }

    /// Takes every queued drawing and hands it to `rec` on the calling
    /// thread, without rows: what the viewer thread does, for a test that
    /// reads back what a producer drew.
    #[cfg(test)]
    pub fn draw_queued(&self, rec: &RecordingStream) {
        while let Some(env) = self.q.try_pop() {
            draw(rec, env.item.entries);
        }
    }
}

/// A stage's handle on [`VIEWER_EDGE`]. It takes the calls a
/// `RecordingStream` takes, and [`Canvas::send`] queues what they drew.
///
/// Owned by one stage and used on its thread: the timelines and the pending
/// drawing are the stage's own, as a `RecordingStream`'s timelines are its
/// thread's.
pub struct Canvas {
    viewer: Arc<ViewerQueue>,
    stage: &'static str,
    time: Cell<Stamps>,
    pending: RefCell<Vec<Entry>>,
}

impl Canvas {
    pub fn set_timestamp_nanos_since_epoch(&self, timeline: &'static str, ns: i64) {
        self.stamp(timeline, Time::TimestampNs(ns));
    }

    pub fn set_duration_secs(&self, timeline: &'static str, secs: f64) {
        self.stamp(timeline, Time::DurationSecs(secs));
    }

    pub fn set_time_sequence(&self, timeline: &'static str, seq: i64) {
        self.stamp(timeline, Time::Sequence(seq));
    }

    pub fn disable_timeline(&self, timeline: &'static str) {
        let mut t = self.time.get();
        t.unset(timeline);
        self.time.set(t);
    }

    fn stamp(&self, timeline: &'static str, v: Time) {
        let mut t = self.time.get();
        t.set(timeline, v);
        self.time.set(t);
    }

    /// Adds `as_components` at the timelines in force to the drawing
    /// [`Canvas::send`] will queue. The SDK's result type, so the drawing
    /// code reads as it did; keeping an entry cannot fail, and a drawing the
    /// viewer never takes is a row on [`VIEWER_EDGE`], not an error here.
    pub fn log<AS: ?Sized + AsComponents>(
        &self,
        path: impl Into<EntityPath>,
        as_components: &AS,
    ) -> RecordingStreamResult<()> {
        self.keep(path.into(), Some(self.time.get()), as_components);
        Ok(())
    }

    /// As [`Canvas::log`], with no timelines: static data.
    pub fn log_static<AS: ?Sized + AsComponents>(
        &self,
        path: impl Into<EntityPath>,
        as_components: &AS,
    ) -> RecordingStreamResult<()> {
        self.keep(path.into(), None, as_components);
        Ok(())
    }

    fn keep<AS: ?Sized + AsComponents>(
        &self,
        path: EntityPath,
        time: Option<Stamps>,
        as_components: &AS,
    ) {
        let batches = as_components.as_serialized_batches();
        if !batches.is_empty() {
            self.pending.borrow_mut().push(Entry {
                path,
                time,
                batches,
            });
        }
    }

    /// Queues everything drawn since the last send as one drawing of
    /// `subject`. Never waits. Nothing drawn, nothing queued.
    pub fn send(&self, subject: Subject) {
        let entries = std::mem::take(&mut *self.pending.borrow_mut());
        if entries.is_empty() {
            return;
        }
        let bytes = entries
            .iter()
            .flat_map(|e| &e.batches)
            .map(|b| b.array.get_buffer_memory_size() as u64)
            .sum();
        self.viewer.push(Drawing {
            subject,
            stage: self.stage,
            bytes,
            entries,
        });
    }

    /// Closes [`VIEWER_EDGE`]: for the last producer, once it has drawn its
    /// last.
    pub fn close(&self) {
        self.viewer.close();
    }
}

impl Drop for Canvas {
    fn drop(&mut self) {
        // A drawing never sent is a picture silently lost; every drawing
        // block ends in `send`. Not checked while a panic unwinds the stage.
        debug_assert!(
            self.pending.borrow().is_empty() || std::thread::panicking(),
            "{}: a drawing was never sent",
            self.stage
        );
    }
}

/// Hands one drawing's entries to the SDK, each at its own timelines. Returns
/// how many it refused.
fn draw(rec: &RecordingStream, entries: Vec<Entry>) -> u64 {
    let mut refused = 0;
    for e in entries {
        let r = match e.time {
            Some(t) => {
                t.apply(rec);
                rec.log_serialized_batches(e.path, false, e.batches)
            }
            None => rec.log_serialized_batches(e.path, true, e.batches),
        };
        refused += u64::from(r.is_err());
    }
    refused
}

/// What the `viewer` thread returns.
#[derive(Debug, Default)]
pub struct ViewerReport {
    /// Drawings the SDK took.
    pub delivered: u64,
    /// `log` calls the SDK refused. rerun 0.38.1 refuses none.
    pub log_errors: u64,
}

/// Body of the `viewer` thread: the only code that calls `log`.
///
/// Pops a drawing, hands it to the SDK -- which blocks here, and only here,
/// while the viewer does not take data -- and writes its row, then the rows of
/// any drawings pushes have lost meanwhile. Returns the stream, which
/// `run::finish` flushes.
pub fn viewer_thread(
    viewer: Arc<ViewerQueue>,
    rec: RecordingStream,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
) -> (ViewerReport, RecordingStream) {
    set_stage_slot(SLOT_VIEWER);
    let rc = ctx.row_ctx();
    let mut report = ViewerReport::default();
    while let Some(env) = viewer.q.pop() {
        let dequeued = now();
        let Queued {
            item,
            enqueued,
            depth_at_push,
            depth_after_pop,
        } = env;
        let Drawing {
            subject,
            bytes,
            entries,
            ..
        } = item;
        let proc_start = now();
        let b0 = bytes_alloc(SLOT_VIEWER);
        report.log_errors += draw(&rec, entries);
        let alloc = bytes_alloc(SLOT_VIEWER) - b0;
        let proc_end = now();
        let mut row = subject.row(&rc, VIEWER_STAGE, bytes);
        row.enqueued_ns = Some(enqueued - rc.t0_host);
        row.dequeued_ns = Some(dequeued - rc.t0_host);
        row.proc_start_ns = Some(proc_start - rc.t0_host);
        row.proc_end_ns = Some(proc_end - rc.t0_host);
        row.queue_wait_ns = Some(dequeued - enqueued);
        row.measurement_age_ns = subject.due_ns.map(|d| proc_end - rc.t0_host - d);
        row.depth_at_push = Some(depth_at_push);
        row.depth_after_pop = depth_after_pop;
        row.bytes_alloc = alloc;
        sink.send(EvRow::Evidence(row), &ctx, VIEWER_STAGE);
        report.delivered += 1;
        write_lost(&viewer, &rc, &sink, &ctx);
    }
    write_lost(&viewer, &rc, &sink, &ctx);
    (report, rec)
}

/// The rows of the drawings pushes have lost since the last call: on
/// [`VIEWER_EDGE`], under the stage whose picture it was.
fn write_lost(viewer: &ViewerQueue, rc: &RowCtx, sink: &EvidenceSink, ctx: &RunCtx) {
    let lost = std::mem::take(&mut *lock(&viewer.lost));
    for l in lost {
        let mut row = l.subject.row(rc, l.stage, l.bytes);
        row.outcome = l.outcome;
        row.reason = l.reason;
        row.enqueued_ns = Some(l.enqueued - rc.t0_host);
        row.depth_at_push = Some(l.depth_at_push);
        row.push_blocked_ns = Some(l.push_blocked_ns);
        sink.send(EvRow::Evidence(row), ctx, VIEWER_STAGE);
    }
}

/// Test hook `--viewer-delay-ms`: the viewer takes nothing from the moment
/// the run's clock starts until `delay` after it, as a viewer frozen for that
/// long does, then everything.
pub struct Freeze {
    delay_ns: i64,
    until: OnceLock<HostTime>,
}

impl Freeze {
    pub fn new(delay: Duration) -> Arc<Self> {
        Arc::new(Freeze {
            delay_ns: i64::try_from(delay.as_nanos()).unwrap_or(i64::MAX),
            until: OnceLock::new(),
        })
    }

    /// The clock has started at `t0`.
    pub fn start(&self, t0: HostTime) {
        let _ = self.until.set(t0 + self.delay_ns);
    }

    fn wait(&self) {
        if let Some(&until) = self.until.get() {
            let rem = until - now();
            if rem > 0 {
                std::thread::sleep(Duration::from_nanos(u64::try_from(rem).unwrap_or(0)));
            }
        }
    }
}

/// A sink that waits out a [`Freeze`] before it passes a message on.
struct FrozenSink {
    inner: Box<dyn LogSink>,
    freeze: Arc<Freeze>,
}

impl LogSink for FrozenSink {
    fn send(&self, msg: LogMsg) {
        self.freeze.wait();
        self.inner.send(msg);
    }

    fn send_all(&self, messages: Vec<LogMsg>) {
        self.freeze.wait();
        self.inner.send_all(messages);
    }

    fn flush_blocking(&self, timeout: Duration) -> Result<(), SinkFlushError> {
        self.inner.flush_blocking(timeout)
    }

    fn defers_finalization_to_shutdown(&self) -> bool {
        self.inner.defers_finalization_to_shutdown()
    }
}

/// The SDK's buffers under [`Freeze`]: 256 KiB in flight instead of 100 MiB,
/// so a frozen sink blocks `log` within a few of the fixture's samples, as
/// the real viewer's blocks it within about six seconds of drive 0005.
const FROZEN_BATCHER: ChunkBatcherConfig = ChunkBatcherConfig {
    max_bytes_in_flight: 256 * 1024,
    ..ChunkBatcherConfig::LOW_LATENCY
};

/// `path` as an `.rrd` written through a [`FrozenSink`]: the recording
/// `--viewer-delay-ms` gives a run.
pub fn frozen_file(path: &Path, freeze: Arc<Freeze>) -> RecordingStreamResult<RecordingStream> {
    let file = FileSink::new(path)?;
    let sink: Box<dyn LogSink> = Box::new(FrozenSink {
        inner: Box::new(file),
        freeze,
    });
    RecordingStreamBuilder::new("pipes")
        .batcher_config(FROZEN_BATCHER)
        .set_sinks(vec![sink])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::buffer::Buffer;
    use pipes_core::clock::{SensorTime, Tov};
    use pipes_core::sample::StreamId;
    use rerun::archetypes::{Image, Scalars};
    use rerun::components::ImageBuffer;
    use rerun::datatypes::{Blob, ChannelDatatype, ColorModel};

    use super::*;

    fn rc() -> RowCtx {
        RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(1_000),
            epoch: 0,
        }
    }

    fn subject(seq: u64) -> Subject {
        let s = Sample {
            stream: StreamId::LIDAR,
            seq,
            arrival_seq: 10 + seq,
            parent: None,
            tov: Tov::Time(SensorTime(5)),
            epoch: 0,
            due: Some(HostTime(1_500)),
            arrival: HostTime(1_600),
            payload: arrow::array::RecordBatch::new_empty(Arc::new(
                arrow::datatypes::Schema::empty(),
            )),
            storage_id: 7,
            decode_ns: 0,
        };
        Subject::of_sample(&rc(), &s)
    }

    /// The camera frame's pixels reach the queue by reference: the serialized
    /// image's buffer is the frame's own, as it was when a stage handed the
    /// image to `log` itself. Queueing a picture costs no copy of it.
    #[test]
    fn a_queued_image_shares_the_frame_buffer() {
        let pixels = Buffer::from_vec(vec![7u8; 4 * 2 * 3]);
        let v = ViewerQueue::new(4);
        let c = v.canvas("rerun");
        let image = Image::from_color_model_and_bytes(
            ImageBuffer(Blob::from(pixels.clone())),
            [4, 2],
            ColorModel::RGB,
            ChannelDatatype::U8,
        );
        c.log("camera/image", &image).unwrap();
        c.send(subject(0));
        let d = v.q.try_pop().unwrap().item;
        let shared = d.entries[0].batches.iter().any(|b| {
            let data = b.array.to_data();
            std::iter::once(&data)
                .chain(data.child_data())
                .flat_map(|a| a.buffers())
                .any(|buf| buf.as_ptr() == pixels.as_ptr())
        });
        assert!(
            shared,
            "the queued image does not point at the frame's buffer"
        );
        assert!(d.bytes >= pixels.len() as u64, "{} B carried", d.bytes);
    }

    /// A full queue drops its OLDEST drawing, the push returns at once, and
    /// the rows balance: one per drawing, delivered or evicted.
    #[test]
    fn a_full_queue_evicts_the_oldest_drawing_and_every_drawing_is_counted() {
        let v = ViewerQueue::new(2);
        let c = v.canvas("cloud");
        for seq in 0..5 {
            c.set_time_sequence("seq", seq);
            c.log("lidar/sweep", &Scalars::single(1.0)).unwrap();
            c.send(subject(seq as u64));
        }
        // Nothing drawn, nothing queued.
        c.send(subject(9));
        assert_eq!((v.admitted(), v.dropped()), (5, 3));
        let lost: Vec<(u64, &str)> = lock(&v.lost)
            .iter()
            .map(|l| (l.subject.seq, l.reason))
            .collect();
        assert_eq!(lost, vec![(0, "evicted"), (1, "evicted"), (2, "evicted")]);
        let kept: Vec<u64> = std::iter::from_fn(|| v.q.try_pop())
            .map(|e| e.item.subject.seq)
            .collect();
        assert_eq!(kept, vec![3, 4]);
        assert_eq!(kept.len() as u64 + v.dropped(), v.admitted());
        assert_eq!(v.push_ns().len(), 5);
    }

    /// A canvas keeps the timelines a `RecordingStream` would, and each entry
    /// carries the ones in force when it was logged.
    #[test]
    fn each_entry_carries_the_timelines_in_force_when_it_was_logged() {
        let v = ViewerQueue::new(4);
        let c = v.canvas("rec");
        c.set_timestamp_nanos_since_epoch("sensor_time", 5);
        c.set_duration_secs("host", 0.5);
        c.set_time_sequence("seq", 3);
        c.log("a", &Scalars::single(1.0)).unwrap();
        c.disable_timeline("sensor_time");
        c.set_time_sequence("seq", 4);
        c.log("b", &Scalars::single(2.0)).unwrap();
        c.log_static("c", &Scalars::single(3.0)).unwrap();
        c.send(subject(0));
        let d = v.q.try_pop().unwrap().item;
        let times: Vec<Option<Stamps>> = d.entries.iter().map(|e| e.time).collect();
        let mut first = Stamps::default();
        first.set("sensor_time", Time::TimestampNs(5));
        first.set("host", Time::DurationSecs(0.5));
        first.set("seq", Time::Sequence(3));
        let mut second = first;
        second.unset("sensor_time");
        second.set("seq", Time::Sequence(4));
        assert_eq!(times, vec![Some(first), Some(second), None]);
    }
}
