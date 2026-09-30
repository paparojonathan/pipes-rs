//! The evidence channel (spec G.1/G.2): every stage `send`s rows into one
//! bounded queue (`evidence->rec`, cap 4096, `Block { 1 s }`) and the `rec`
//! thread drains it to `evidence.csv` / `events.csv`. On a Block timeout the
//! channel degrades to DropOldest once, records exactly one
//! `RecorderDegraded` event and counts lost rows: the pipeline never stalls
//! on its recorder for longer than `max_wait` (design §7).
//!
//! M8: with `--dashboard` the same thread also mirrors each row onto the run's
//! Rerun recording as scalars ([`Dashboard`]) — a READER of the evidence
//! stream, so the live view costs no new instrumentation anywhere else. It
//! draws on a [`Canvas`] like every stage: its drawings go through the
//! viewer's own queue, so a slow viewer can never back up into the evidence
//! channel and, through it, into every stage that writes rows.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pipes_core::alloc::{set_stage_slot, SLOT_REC};
use pipes_core::clock::now;
use pipes_core::evidence::{Event, EventKind, Evidence, Outcome};
use pipes_core::queue::{BoundedQueue, PushOutcome, QueuePolicy};
use pipes_core::sample::StreamId;
use pipes_kitti::layout::ABSENT_IN_SOURCE;
use rerun::archetypes::{
    Clear, GraphEdges, GraphNodes, Measurements, Scalars, SeriesLines, SeriesPoints, StateChange,
    StateConfiguration, TextDocument, TextLog,
};
use rerun::components::{Color, InterpolationMode, MarkerShape, MediaType, TextLogLevel};
use rerun::{AsComponents, RecordingStream};

use crate::consumers::{log_no_frame, log_no_sweep};
use crate::dashboard::{
    entity, lane_path, queue_lane, send_blueprint, Mode, ReadyDot, StageClock, ADMISSION_EDGES,
    CHAIN_BYTES, EDGES, GRAPH_EDGES, GRAPH_NODES, LANE_CAMERA_NAME, LANE_LIDAR_NAME, LANE_NAMES,
    LANE_PAIRING_NAME, PAIRING_DONE_TOP, READY_DOTS, SWEEP_END_LEAF, SWEEP_END_ZOOM,
};
use crate::run::RunCtx;
use crate::viewer::{Canvas, Subject, VIEWER_EDGE};

/// What travels on `evidence->rec`.
///
/// `Evidence` is ~300 bytes and `Event` ~90, which trips
/// `clippy::large_enum_variant`. Its suggested fix — box the large variant —
/// is the wrong trade here: every stage sends one of these per sample per
/// edge, so a `Box` would put a heap allocation on the evidence path of the
/// pipeline whose allocations this project exists to count. The cost of not
/// boxing is 4096 × ~300 B ≈ 1.2 MB in the channel's buffer, paid once,
/// before the clock starts.
#[allow(clippy::large_enum_variant)]
pub enum EvRow {
    Evidence(Evidence),
    Event(Event),
}

/// The producer side of `evidence->rec`, shared by every stage as `Arc<EvidenceSink>`.
pub struct EvidenceSink {
    q: Arc<BoundedQueue<EvRow>>,
    degraded: AtomicBool,
    lost: AtomicU64,
    blocked_ns: Mutex<Vec<i64>>,
}

impl EvidenceSink {
    pub fn new(cap: usize, max_wait: Duration) -> Self {
        EvidenceSink {
            q: Arc::new(BoundedQueue::new(cap, QueuePolicy::Block { max_wait })),
            degraded: AtomicBool::new(false),
            lost: AtomicU64::new(0),
            blocked_ns: Mutex::new(Vec::new()),
        }
    }

    /// Pushes one row: bounded wait while the channel is healthy, never a
    /// wait once degraded. `stage` names the caller for the degrade event.
    pub fn send(&self, row: EvRow, ctx: &RunCtx, stage: &'static str) {
        let arrival_seq_after = match &row {
            // A driver `Missing` row carries no admission order, so a degrade
            // event triggered by one is anchored at 0. That is imprecise, not
            // ambiguous: `Event::arrival_seq_after` is still a plain `u64`, and
            // making it `Option` too is a follow-up that also touches
            // events.csv's schema.
            EvRow::Evidence(e) => e.arrival_seq.unwrap_or(0),
            EvRow::Event(ev) => ev.arrival_seq_after,
        };
        let t0 = now();
        let out = self.q.push(row, t0);
        self.blocked_ns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(now() - t0);
        match out {
            PushOutcome::Accepted { .. } => {}
            PushOutcome::Timeout(env) => {
                let first = !self.degraded.swap(true, Relaxed);
                if first {
                    self.q.set_policy(QueuePolicy::DropOldest);
                }
                if !matches!(
                    self.q.try_push(env.item, now()),
                    PushOutcome::Accepted { .. }
                ) {
                    self.lost.fetch_add(1, Relaxed);
                }
                if first {
                    let ev =
                        Event::recorder_degraded(&ctx.row_ctx(), now(), stage, arrival_seq_after);
                    if !matches!(
                        self.q.try_push(EvRow::Event(ev), now()),
                        PushOutcome::Accepted { .. }
                    ) {
                        self.lost.fetch_add(1, Relaxed);
                    }
                }
            }
            PushOutcome::Evicted(_) | PushOutcome::Rejected(_) | PushOutcome::Closed(_) => {
                self.lost.fetch_add(1, Relaxed);
            }
        }
    }

    pub fn close(&self) {
        self.q.close();
    }

    pub fn queue(&self) -> Arc<BoundedQueue<EvRow>> {
        Arc::clone(&self.q)
    }

    /// Rows that never reached the recorder.
    pub fn lost(&self) -> u64 {
        self.lost.load(Relaxed)
    }

    pub fn degraded(&self) -> bool {
        self.degraded.load(Relaxed)
    }

    /// Duration of every `send` so far (ns), for the `push_blocked_ns p99` line.
    pub fn blocked_ns(&self) -> Vec<i64> {
        self.blocked_ns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

// The palette groups by MEANING, not by variety, because a dozen entities in
// one plot is unreadable when every colour is arbitrary: a series is the
// colour of what it is about -- the camera's blue, the lidar's teal, the
// detection purple, the answer's gold -- anything that counts a failure is
// red, and a reference bound is thin and grey so it reads as a backdrop
// rather than a measurement.
/// The detection family, #9B7BD8: the detections, tracks and state on the
/// pipeline graph.
const C_DETECTION: [u8; 3] = [155, 123, 216];
/// A failure that has already happened, #E5484D -- and the time to contact,
/// the one number on the demo screen that is bad when it is small.
const C_FAIL: [u8; 3] = [229, 72, 77];
/// A band a measurement is read inside, and what belongs to no sensor, the
/// admission on the pipeline graph: mid grey. The viewer draws a band only
/// in an opaque colour at a width of at least 1.
const C_BOUND: [u8; 3] = [150, 150, 150];
/// A reference line a measurement is read against -- the 3 s warning, a
/// sweep's ends and its trigger: a bright grey, #D2D2D2, because a mid-grey
/// line was invisible on the viewer's black.
const C_REFERENCE: [u8; 3] = [210, 210, 210];
/// Something was missed and the pipeline carried on: a lane's `skipping` or
/// `stale`, and the answer box on a frame that came from a stale pair. `pub`
/// for the box, on the terms [`C_ANSWER`] is.
pub const C_ATTENTION: [u8; 3] = [240, 170, 32];
/// Healthy: a lane's `ok` or `same`, #3C8250 -- a darker green than the
/// fused population's, so a lane of health reads as a backdrop.
const C_LANE_OK: [u8; 3] = [60, 130, 80];
/// Nothing there at all: a lane's `no sweep`, `no frame` or `source gap`, for
/// a frame a sensor's source never had (`absent_in_source`), #808080. Grey,
/// because it is neither the pipeline's health nor its failure -- nothing
/// upstream dropped it -- and a reader must not take the source's gap for a
/// queue's loss. The WARN log and the headline say why.
const C_ABSENT: [u8; 3] = [128, 128, 128];
/// The ANSWER's own colour, gold, and deliberately a colour nothing else in
/// this palette uses: the answer's box, its label and its 3D wireframe, and
/// the ready dot and graph node that time it. Every other colour here marks
/// the rest of the PIPELINE -- how late, how many bytes -- or a sensor. Its
/// distance and closing speed are words in its label and the headline
/// rather than lines on a plot.
///
/// `pub` because `consumers` draws the answer's rectangle over the camera
/// image in this colour. Exported rather than repeated: two constants that
/// have to agree are two constants that will one day not, and a test that they
/// still match is a worse guarantee than there being only one of them.
pub const C_ANSWER: [u8; 3] = [255, 208, 48];
// The fusion's three populations, in the dashboard layout's own palette
// (v2 spec: fused #5CB278, lidar #00A8B0, camera #408CD6). Colour means
// PROVENANCE here -- which sensor vouches for the object -- never identity:
// the per-id hues the tracks used to wear said nothing a reader could act on.
/// A track the camera confirmed: green, the one population both sensors see.
pub const C_FUSED: [u8; 3] = [92, 178, 120];
/// A track only the lidar holds: teal, the lidar's colour everywhere.
pub const C_LIDAR_ONLY: [u8; 3] = [0, 168, 176];
/// `reduce`'s ready dot, the lidar's first stage: a lighter voice of the
/// lidar's teal, #80D2D7, so it reads as the lidar's and apart from
/// `detect`'s, which is the lidar's half.
const C_LIDAR_EARLY: [u8; 3] = [128, 210, 215];
/// A detection only the camera holds: blue, the camera's colour everywhere.
pub const C_CAMERA_ONLY: [u8; 3] = [64, 140, 214];

/// How one series is marked on its plot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mark {
    /// A line. Counts hold their value until the next sample (`step`); a
    /// continuous measurement is interpolated between samples.
    Line { step: bool },
    /// One marker per sample, never joined: a series whose consecutive
    /// samples need not describe the same thing, which a line would join into
    /// a jump that no object made.
    Points(MarkerShape),
}

/// Legend name, colour and mark of one scalar series.
struct SeriesStyle {
    /// The legend entry, complete: the quantity and its unit, and nothing
    /// about which edge it is on -- the view's title says that, and a legend
    /// of eleven "cam0->proc queue wait (ms)"-style entries was unreadable.
    label: &'static str,
    rgb: [u8; 3],
    /// The line's width, or the marker's size.
    width: f32,
    mark: Mark,
}

/// The longest legend entry a plot legend shows without eating the plot.
#[cfg(test)]
const LABEL_MAX: usize = 20;

impl SeriesStyle {
    /// Whether the series is a line that holds its value between samples.
    #[cfg(test)]
    fn step(&self) -> bool {
        self.mark == Mark::Line { step: true }
    }

    /// The static archetype that names, colours and marks one entity's
    /// series: `SeriesLines` for a line, `SeriesPoints` for markers.
    fn archetype(&self) -> Box<dyn AsComponents> {
        let color = Color::from_rgb(self.rgb[0], self.rgb[1], self.rgb[2]);
        match self.mark {
            Mark::Line { step } => Box::new(
                SeriesLines::new()
                    .with_names([self.label])
                    .with_colors([color])
                    .with_widths([self.width])
                    .with_interpolation_mode(if step {
                        InterpolationMode::StepAfter
                    } else {
                        InterpolationMode::Linear
                    }),
            ),
            Mark::Points(marker) => Box::new(
                SeriesPoints::new()
                    .with_names([self.label])
                    .with_colors([color])
                    .with_markers([marker])
                    .with_marker_sizes([self.width]),
            ),
        }
    }
}

/// Every scalar leaf the run can emit. `series_style` must answer for all of
/// them (pinned by a test), so adding a series without giving it a name and a
/// colour fails the build's tests rather than quietly producing another
/// anonymous line in the viewer.
#[cfg(test)]
const SERIES_LEAVES: [&str; 15] = [
    // Logged by the `track` stage straight onto the recording, for every
    // sweep: the pairing window is in no evidence column.
    "sweep_start",
    "sweep_end",
    "trigger",
    "camera",
    // On the same plot, and on its zoom onto the sweep's end, by the
    // evidence mirror ([`Dashboard::log_done`]): when the pairing's two
    // halves, the lidar's first stage and the answer were ready. The zoom's
    // line at the sweep's end is the `sweep_end` above.
    "1_camera_done",
    "2_reduce_done",
    "3_detect_done",
    "4_answer_done",
    // The answer's own series. Logged by `state-sink` straight onto the
    // recording rather than mirrored off an evidence row -- an evidence row
    // carries how long a sample took and how big it was, never what it SAID.
    // They are in this table for the reason everything else is: an unstyled
    // series is an anonymous line in somebody's viewer.
    "ttc_s",
    "ttc_warn",
    // The fusion's populations in the camera frame, logged by `state-sink`
    // off the answer's records and its `camera_only` column, on every answer.
    "fused",
    "lidar_only",
    "camera_only",
    // Beside them, the fused share of each sensor's objects in the frame.
    "fused_fraction",
    "camera_fused_fraction",
];

/// The static style of a grey band -- a `Measurements` series, whose one
/// standard deviation either side of its value is drawn as a translucent band
/// -- named `name`, in ms. Opaque grey and a 1 px line: the viewer drew
/// nothing for a translucent colour at width 0.
pub fn band_archetype(name: &str) -> Measurements {
    Measurements::update_fields()
        .with_names([name])
        .with_colors([Color::from_rgb(C_BOUND[0], C_BOUND[1], C_BOUND[2])])
        .with_widths([1.0f32])
        .with_units(["ms"])
}

/// The static styling archetype of one scalar leaf, for a stage that logs its
/// own series directly onto the recording instead of through the evidence
/// mirror.
///
/// `None` for a leaf with no style, which is a bug the test over
/// [`SERIES_LEAVES`] exists to catch before it ships as an unnamed grey line.
pub fn series_archetype(leaf: &str) -> Option<Box<dyn AsComponents>> {
    series_style(leaf).map(|s| s.archetype())
}

/// How to draw one scalar leaf, or `None` for a leaf with no style yet.
fn series_style(leaf: &str) -> Option<SeriesStyle> {
    let s = |label, rgb, width, step| {
        Some(SeriesStyle {
            label,
            rgb,
            width,
            mark: Mark::Line { step },
        })
    };
    let points = |label, rgb, size, marker| {
        Some(SeriesStyle {
            label,
            rgb,
            width: size,
            mark: Mark::Points(marker),
        })
    };
    match leaf {
        // The pairing window, in ms from the sweep's start: its two ends as
        // bright reference lines (the grey band between them alone is faint
        // on black), the trigger as a thinner one, and the paired camera
        // frame's instant as a POINT per sweep, in the camera's blue. Inside
        // the band it is contemporaneous with the sweep; below it, stale.
        "sweep_start" => s("sweep start (ms)", C_REFERENCE, 1.5, true),
        "sweep_end" => s("sweep end (ms)", C_REFERENCE, 1.5, true),
        "trigger" => s("trigger (ms)", C_REFERENCE, 1.0, true),
        "camera" => points("camera frame (ms)", C_CAMERA_ONLY, 2.5, MarkerShape::Circle),
        // When the pairing's halves and its answer were ready, on the same
        // plot and on its zoom, as a DIAMOND each -- a stage finishing, where
        // the circle is an instant measured -- in the colour of what it is
        // about: the camera's blue, the lidar's teal (lighter for `reduce`,
        // its first stage), the answer's gold. A blue diamond above
        // `detect`'s is a camera frame the lidar's boxes waited for.
        "1_camera_done" => points("camera done (ms)", C_CAMERA_ONLY, 3.5, MarkerShape::Diamond),
        "2_reduce_done" => points("reduce done (ms)", C_LIDAR_EARLY, 3.5, MarkerShape::Diamond),
        "3_detect_done" => points("detect done (ms)", C_LIDAR_ONLY, 3.5, MarkerShape::Diamond),
        "4_answer_done" => points("answer done (ms)", C_ANSWER, 3.5, MarkerShape::Diamond),
        // Time to contact with the object the answer flags, as POINTS: the
        // flagged object changes from sweep to sweep, and a line joined one
        // object's approach to the next one's with a vertical jump no object
        // made, where points show each approach as its own diagonal falling
        // through the warning line. Logged only while the object closes, so
        // a receding object is a gap and not a TTC of 0, which reads as a
        // collision. In the warning red: it is the one number on the demo
        // screen a reader should be alarmed by.
        "ttc_s" => points("TTC (s)", C_FAIL, 2.0, MarkerShape::Circle),
        // The line those points are read against: 3 s.
        "ttc_warn" => s("TTC warning (s)", C_REFERENCE, 1.5, true),
        // Counts, so they hold their value: in the frame, how many tracks the
        // camera confirmed, how many only the lidar holds, and how many
        // detections only the camera holds. A stale camera moves the first
        // down and the other two up.
        "fused" => s("fused (count)", C_FUSED, 2.0, true),
        "lidar_only" => s("lidar-only (count)", C_LIDAR_ONLY, 2.0, true),
        "camera_only" => s("camera-only (count)", C_CAMERA_ONLY, 2.0, true),
        // The fused share of each sensor's objects in the frame, held per
        // answer like the counts it divides, in the colour of the sensor
        // whose objects it is a share of: the lidar's tracks the camera
        // confirmed, and the camera's detections the lidar confirmed.
        "fused_fraction" => s("of lidar (share)", C_LIDAR_ONLY, 2.0, true),
        "camera_fused_fraction" => s("of camera (share)", C_CAMERA_ONLY, 2.0, true),
        _ => None,
    }
}

/// The states one lane can show, with a colour each. Logged once, statically,
/// as the lane's `StateConfiguration`, so the viewer draws a green lane for
/// health and a red one for a failure instead of picking colours by first
/// sight.
struct Lane {
    values: &'static [&'static str],
    colors: &'static [[u8; 3]],
}

impl Lane {
    fn archetype(&self) -> StateConfiguration {
        StateConfiguration::new()
            .with_values(self.values.iter().copied())
            .with_labels(self.values.iter().copied())
            .with_colors(
                self.colors
                    .iter()
                    .map(|c| Color::from_rgb(c[0], c[1], c[2])),
            )
    }
}

/// How a queue lane reads: green while every sample gets through, amber
/// (`skipping`) when a delivered sample's predecessor on the edge was never
/// seen, red (`dropping`) when a sample was dropped.
const LANE_QUEUE: Lane = Lane {
    values: &["ok", "skipping", "dropping"],
    colors: &[C_LANE_OK, C_ATTENTION, C_FAIL],
};

/// How the lidar's lane reads: the queue's three states, and grey `no
/// sweep` for a frame the lidar's source never had -- the gap is the source's,
/// not the queue's, and is drawn on the lidar's own lane at its own time.
const LANE_LIDAR: Lane = Lane {
    values: &["ok", "skipping", NO_SWEEP, "dropping"],
    colors: &[C_LANE_OK, C_ATTENTION, C_ABSENT, C_FAIL],
};

/// How the camera's lane reads: the queue's three states, and grey `no
/// frame` for a frame the camera's source never had, on the same terms.
const LANE_CAMERA: Lane = Lane {
    values: &["ok", "skipping", NO_FRAME, "dropping"],
    colors: &[C_LANE_OK, C_ATTENTION, C_ABSENT, C_FAIL],
};

/// How the pairing lane reads: green while the fusion pairs every sweep with
/// its own frame, amber (`stale`) when it took an older frame, red
/// (`expired`) when a sweep's set expired and the answer lost that sweep --
/// `pair_late`, `pair_dropped` or `pair_absent`, which the WARN log names --
/// and grey `source gap` for an instant whose sweep or camera frame a sensor's
/// source never had, so the fusion had nothing to pair.
const LANE_PAIRING: Lane = Lane {
    values: &["ok", "stale", SOURCE_GAP, "expired"],
    colors: &[C_LANE_OK, C_ATTENTION, C_ABSENT, C_FAIL],
};

/// The lidar lane's word for a frame whose sweep is absent in the source.
const NO_SWEEP: &str = "no sweep";

/// The camera lane's word for a frame whose PNG is absent in the source.
const NO_FRAME: &str = "no frame";

/// The pairing lane's word for an instant a gap in either sensor's source
/// left without a partner. Not "absent", which there would read as the
/// camera stream's `pair_absent`.
const SOURCE_GAP: &str = "source gap";

/// Every lane entity the run can emit, for the test that no lane ships
/// without its `StateConfiguration`.
#[cfg(test)]
const LANE_PATHS: [&str; 5] = [
    "lanes/pairing",
    "lanes/camera_queue",
    "lanes/lidar_queue",
    "lanes/detect_queue",
    "lanes/fusion_queue",
];

/// How to draw one lane, by its entity path, or `None` for a path with no
/// lane style, which is the bug the test over [`LANE_PATHS`] exists to catch.
fn lane_style(path: &str) -> Option<&'static Lane> {
    if path == lane_path(LANE_PAIRING_NAME) {
        Some(&LANE_PAIRING)
    } else if path == lane_path(LANE_LIDAR_NAME) {
        Some(&LANE_LIDAR)
    } else if path == lane_path(LANE_CAMERA_NAME) {
        Some(&LANE_CAMERA)
    } else if LANE_NAMES
        .iter()
        .any(|n| *n != LANE_PAIRING_NAME && path == lane_path(n))
    {
        Some(&LANE_QUEUE)
    } else {
        None
    }
}

/// What one row says about a lane: `None` when it is clean, or the lane's
/// word for what went wrong and how bad it is (1 amber, 2 red).
type LaneEvent = Option<(&'static str, u8)>;

/// A queue lane's event for one row: `dropping` for any sample the queue did
/// not deliver, `skipping` for a delivered one whose predecessor on the edge
/// was never seen, clean otherwise.
fn queue_event(outcome: Outcome, gap: Option<f64>) -> LaneEvent {
    match outcome {
        Outcome::Delivered if gap.is_some_and(|g| g > 1.0) => Some(("skipping", 1)),
        Outcome::Delivered => None,
        _ => Some(("dropping", 2)),
    }
}

/// The pairing lane's event for the fusion's own row: `reason` is blank on a
/// clean pair, `stale` on a Delivered row paired inside `--pair-stale-ms`,
/// and names how the set expired (`pair_late`, `pair_dropped`,
/// `pair_absent`, `pair_absent_in_source`) on a Missing one -- the last a gap
/// in the camera's source, not a failure of the pipeline.
fn pairing_event(e: &Evidence) -> LaneEvent {
    match (e.outcome, e.reason) {
        (Outcome::Delivered, "") => None,
        (Outcome::Delivered, _) => Some(("stale", 1)),
        (_, "pair_absent_in_source") => Some((SOURCE_GAP, 2)),
        _ => Some(("expired", 2)),
    }
}

/// How many rows without its failure step a lane down, per edge feeding it:
/// a second of them at 10 Hz.
const LANE_CLEAR_AFTER: u32 = 10;

/// How many rows without its failure step the lane `name` down: a second's
/// worth of ITS rows, [`LANE_CLEAR_AFTER`] for each edge that feeds it. The
/// fusion's lane is fed by two, `cam_det->track` and `obj->track`, and at a
/// flat ten it cleared in half a second: on a run whose fusion input
/// dropped a sweep every second and a half it flickered between dropping
/// and skipping in half-second blocks too short to carry their words.
fn lane_clear_after(name: &str, camera_edge: Option<&str>) -> u32 {
    let edges = EDGES
        .iter()
        .filter(|e| queue_lane(e, camera_edge) == Some(name))
        .count()
        .max(1);
    LANE_CLEAR_AFTER * edges as u32
}

/// One sticky lane: a state that is logged only when it changes.
///
/// A lane that logged every sample's word shattered into grey "N states"
/// blocks wherever the words alternated -- a queue dropping every other frame
/// alternates `dropping` and `skipping` -- so the lane holds its worst recent
/// state instead. A failure as bad as the one showing, or worse, sets it
/// (and restarts its count); a lesser one does not. Once the state showing
/// has not recurred for a second's worth of the lane's rows
/// ([`lane_clear_after`]), the lane steps down to the worst lesser failure
/// of those rows, or back to `ok` if they were all clean. So one expired
/// sweep ahead of a run of stale ones reads `expired` for a second and
/// `stale` after it, not `expired` for the whole run.
#[derive(Debug)]
struct Sticky {
    word: &'static str,
    severity: u8,
    /// Rows since the state showing was last set or confirmed.
    quiet: u32,
    /// Rows of quiet that step the state down.
    clear_after: u32,
    /// The worst lesser failure seen in those rows.
    lesser: Option<(u8, &'static str)>,
    started: bool,
}

impl Default for Sticky {
    fn default() -> Self {
        Sticky::new(LANE_CLEAR_AFTER)
    }
}

impl Sticky {
    fn new(clear_after: u32) -> Self {
        Sticky {
            word: "ok",
            severity: 0,
            quiet: 0,
            clear_after,
            lesser: None,
            started: false,
        }
    }

    /// Feeds one row's event; returns the word to log when the lane's state
    /// changed, or on the lane's first row.
    fn observe(&mut self, event: LaneEvent) -> Option<&'static str> {
        let before = self.word;
        match event {
            Some((word, severity)) if severity >= self.severity => {
                self.word = word;
                self.severity = severity;
                self.quiet = 0;
                self.lesser = None;
            }
            _ => {
                if let Some((word, severity)) = event {
                    if self.lesser.is_none_or(|(s, _)| severity >= s) {
                        self.lesser = Some((severity, word));
                    }
                }
                self.quiet += 1;
                if self.severity > 0 && self.quiet >= self.clear_after {
                    let (severity, word) = self.lesser.take().unwrap_or((0, "ok"));
                    self.word = word;
                    self.severity = severity;
                    self.quiet = 0;
                }
            }
        }
        let first = !self.started;
        self.started = true;
        (first || before != self.word).then_some(self.word)
    }
}

/// Where a ready dot is drawn on the pairing plot, in ms from its sweep's
/// start on the sensor clock: the sweep's own length, `span_ns`, then how
/// long after the sweep was due its stage finished, `after_due_ns`, which is
/// on the host clock and which `--rate` stretches onto the sensor's. Held at
/// [`PAIRING_DONE_TOP`] periods when it is that late or later.
fn done_ms(span_ns: i64, after_due_ns: i64, rate: f64, period_ms: f64) -> f64 {
    let ms = (span_ns as f64 + after_due_ns as f64 * rate) * 1e-6;
    ms.min(PAIRING_DONE_TOP * period_ms)
}

/// Where a ready dot is drawn on the zoom onto the sweep's end, in ms after
/// the sweep ended on the sensor clock: `after_due_ns` stretched by
/// `--rate`, held inside [`SWEEP_END_ZOOM`] periods.
fn zoom_ms(after_due_ns: i64, rate: f64, period_ms: f64) -> f64 {
    let (early, late) = SWEEP_END_ZOOM;
    (after_due_ns as f64 * rate * 1e-6).clamp(early * period_ms, late * period_ms)
}

/// The edges the two tables read, every delivered row of each: the byte
/// chain's links as the next stage read them and as the step before built
/// them ([`CHAIN_BYTES`]), and every stage's own row and what it handed on
/// ([`stage_rows`]) -- the camera stage's under either of its names.
const TABLE_EDGES: [&str; 14] = [
    "velo->reduce",
    "det->detect",
    "obj->track",
    "track->state",
    "state->sink",
    "velo",
    "det",
    "obj",
    "track",
    "state",
    "cam0",
    "cam_det",
    "cam0->camdet",
    "cam0->proc",
];

/// Answers after which the tables are first written, so a live viewer has
/// them to read long before the run ends.
const TABLE_AFTER_ANSWERS: u64 = 10;

/// The pipeline graph's node radius, in the graph's own units.
const GRAPH_NODE_RADIUS: f32 = 6.0;

/// Every delivered row of one of the tables' edges, for their medians: a few
/// hundred per edge on a run.
#[derive(Debug, Default)]
struct EdgeRows {
    /// `payload_bytes`: what the edge carried.
    carried: Vec<u64>,
    /// `bytes_alloc`: what the row's stage allocated.
    alloc: Vec<u64>,
    /// The stage's own time on the row, ns: `proc_end - proc_start` on a
    /// queue's row, `decode_ns` on a sensor driver's.
    time_ns: Vec<i64>,
    /// Rows whose consumer named a buffer other than its producer's.
    copied: u64,
}

/// The median of `v`, the upper one of an even count; `None` for no rows.
fn median<T: Copy + Ord>(v: &[T]) -> Option<T> {
    let mut v = v.to_vec();
    v.sort_unstable();
    v.get(v.len() / 2).copied()
}

/// A byte count the way the tables and the graph print it: `1.97 MB`,
/// `518 kB`, `341 B`, in decimal units.
fn human_bytes(b: u64) -> String {
    let f = b as f64;
    if f >= 1e6 {
        format!("{:.2} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.0} kB", f / 1e3)
    } else {
        format!("{b} B")
    }
}

/// A time the way the stage table prints it, to about two figures: `36 ms`,
/// `4.3 ms`, `0.12 ms`, `0.034 ms`.
fn human_ms(ns: i64) -> String {
    let ms = ns as f64 * 1e-6;
    if ms >= 10.0 {
        format!("{ms:.0} ms")
    } else if ms >= 1.0 {
        format!("{ms:.1} ms")
    } else if ms >= 0.1 {
        format!("{ms:.2} ms")
    } else {
        format!("{ms:.3} ms")
    }
}

/// One row of the byte table: a link ([`crate::dashboard::ChainLink`]), and its medians --
/// what it carried, what its step allocated to build it and what the next
/// stage allocated to read it -- `None` where the run had no such row; how
/// many rows they are over, the samples the link delivered, which on a run
/// that evicted is fewer than the sweeps; and how many of those rows read a
/// copy of their producer's buffer.
#[derive(Clone, Copy)]
struct ByteLink<'a> {
    name: &'a str,
    step: &'a str,
    carried: Option<u64>,
    built: Option<u64>,
    read: Option<u64>,
    rows: usize,
    copied: u64,
}

/// The byte chain as a markdown table -- each link, what the step that made
/// it does, what it carried, the change from the link before (a shrink as
/// `÷ 3.8`, a growth as `× 1.4`), what the step allocated to build it and
/// what the next stage allocated to read it -- then what those two columns
/// mean, whether every link was read in place, and what the medians are
/// over: one number when every link delivered the lidar driver's `sweeps`,
/// each link's own count when they did not, so a run that evicted does not
/// read as a median of sweeps a link never saw.
fn byte_table(links: &[ByteLink], sweeps: usize) -> String {
    let show = |b: Option<u64>| b.map_or_else(|| "-".to_string(), human_bytes);
    let mut out = String::from(
        "| link | what the step does | carried | change | built | read |\n\
         |---|---|---:|---:|---:|---:|\n",
    );
    let mut prev: Option<u64> = None;
    for (i, l) in links.iter().enumerate() {
        let change = match (prev, l.carried) {
            (Some(a), Some(b)) if a > 0 && b > 0 && b <= a => {
                format!("÷ {:.1}", a as f64 / b as f64)
            }
            (Some(a), Some(b)) if a > 0 && b > a => format!("× {:.1}", b as f64 / a as f64),
            _ => String::new(),
        };
        out.push_str(&format!(
            "| {} {} | {} | **{}** | {} | {} | {} |\n",
            i + 1,
            l.name,
            l.step,
            show(l.carried),
            change,
            show(l.built),
            show(l.read)
        ));
        prev = l.carried;
    }
    out.push_str(
        "\n**built** is what the step allocated to write its result: the one buffer that result \
         lives in, handed on by reference. **read** is what the next stage allocated to read it: \
         0 B is in place, no copy.\n\n",
    );
    let not_in_place: Vec<String> = links
        .iter()
        .enumerate()
        .filter(|(_, l)| l.read != Some(0) || l.copied > 0)
        .map(|(i, l)| format!("{} {}", i + 1, l.name))
        .collect();
    if not_in_place.is_empty() {
        out.push_str(
            "Every link was read in place, the same `storage_id` at both ends of every row: \
             the lidar driver's, from the file into Arrow, is the chain's one copy.",
        );
    } else {
        out.push_str(&format!("Not read in place: {}.", not_in_place.join(", ")));
    }
    let rows: Vec<usize> = links.iter().map(|l| l.rows).collect();
    if rows.iter().all(|&n| n == sweeps) {
        out.push_str(&format!(" Medians of {sweeps} sweeps."));
    } else {
        let each: Vec<String> = rows.iter().map(usize::to_string).collect();
        out.push_str(&format!(
            " Medians of each link's own rows: {} for links 1 to {}, of the driver's {sweeps} sweeps.",
            each.join(", "),
            rows.len()
        ));
    }
    out
}

/// One row of the stage table: a stage, what it does, the edge whose rows
/// time it, and the producer row whose payload it hands on, with what that
/// payload is.
#[derive(Clone, Copy, Debug)]
struct StageRow {
    stage: &'static str,
    what: &'static str,
    /// The stage's own row: a queue's, whose `proc_end - proc_start` times
    /// the stage, or a sensor driver's, whose `decode_ns` does.
    timed_on: &'static str,
    hands_on: &'static str,
    noun: &'static str,
}

/// Every stage the table can list, the camera's, the lidar's and the
/// fusion's: the camera stage is the detector with it and `proc` without,
/// whichever feeds the answer (`camera_edge`), and it is second.
fn stage_rows(camera_edge: Option<&str>) -> [StageRow; 7] {
    let camera = if camera_edge == Some("cam0->proc") {
        StageRow {
            stage: "proc",
            what: "grayscale pass, the camera's half of each pair",
            timed_on: "cam0->proc",
            hands_on: "cam_det",
            noun: "frame reference",
        }
    } else {
        StageRow {
            stage: "camdet",
            what: "finds objects with YOLOX-Nano, on one thread",
            timed_on: "cam0->camdet",
            hands_on: "cam_det",
            noun: "of boxes",
        }
    };
    [
        StageRow {
            stage: "cam0",
            what: "decodes each PNG into an Arrow buffer",
            timed_on: "cam0",
            hands_on: "cam0",
            noun: "image",
        },
        camera,
        StageRow {
            stage: "velo",
            what: "loads each 360° sweep: x, y, z, reflectance",
            timed_on: "velo",
            hands_on: "velo",
            noun: "sweep",
        },
        StageRow {
            stage: "reduce",
            what: "averages the points into 20 cm cubes",
            timed_on: "velo->reduce",
            hands_on: "det",
            noun: "of cubes",
        },
        StageRow {
            stage: "detect",
            what: "removes the ground, boxes each cluster",
            timed_on: "det->detect",
            hands_on: "obj",
            noun: "of boxes",
        },
        StageRow {
            stage: "track",
            what: "gives boxes ids, fuses them with the camera",
            timed_on: "obj->track",
            hands_on: "track",
            noun: "of tracks",
        },
        StageRow {
            stage: "state",
            what: "writes the answer, flags the nearest in path",
            timed_on: "track->state",
            hands_on: "state",
            noun: "answer",
        },
    ]
}

/// One stage's line of the stage table: its row, and its medians.
struct StageLine<'a> {
    row: &'a StageRow,
    time_ns: Option<i64>,
    carried: Option<u64>,
}

/// The stages as a markdown table -- each stage, what it does, its median
/// time per sample and what it hands on -- then `notes`, one paragraph each.
fn stage_table(lines: &[StageLine], notes: &[String]) -> String {
    let mut out = String::from("| stage | what it does | time | hands on |\n|---|---|---:|---|\n");
    for l in lines {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            l.row.stage,
            l.row.what,
            l.time_ns.map_or_else(|| "-".to_string(), human_ms),
            l.carried.map_or_else(
                || "-".to_string(),
                |b| format!("{} {}", human_bytes(b), l.row.noun)
            ),
        ));
    }
    for n in notes {
        out.push('\n');
        out.push_str(n);
        out.push('\n');
    }
    out
}

/// What the camera stage's node on the pipeline graph says it does: the
/// detector's boxes, or `proc`'s grayscale frame reference.
fn camera_words(stage: &str) -> &'static str {
    if stage == "proc" {
        "grayscale, frame ref"
    } else {
        "YOLOX-Nano boxes"
    }
}

/// How many frames behind the newest a half-joined sweep is kept: one whose
/// other half has not come in by then never will -- its camera frame was
/// dropped, or a source had none.
const HALVES_KEPT: u64 = 32;

/// The two halves of one sweep, by its frame number, until both are in:
/// when the sweep was due and how long it was (its driver's row), when the
/// camera stage finished with the frame of the same number, and when
/// `detect` finished with the sweep. The camera's own row cannot place it
/// against the sweep -- its `due` is the frame's instant, mid-sweep -- so
/// the camera's dot waits here for the sweep's due, and whether it came
/// after the lidar waits for `detect`.
#[derive(Debug, Default)]
struct Halves {
    /// The sweep's due and its length, ns.
    sweep: Option<(i64, i64)>,
    camera: Option<i64>,
    lidar: Option<i64>,
    /// The camera's dot is drawn.
    drawn: bool,
    /// The two finishes are counted.
    compared: bool,
}

/// A sensor whose source can lack a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Sensor {
    Camera,
    Lidar,
}

/// The sensors whose frame numbers a stream's `seq` is: the sensor's own
/// stream, and the streams derived from it that carry the source frame's
/// number. A frame absent in the source of one of them is a frame the stream
/// never had to deliver, on every edge it crosses.
fn sensors_of(stream: u8) -> &'static [Sensor] {
    match StreamId(stream) {
        StreamId::CAM0 | StreamId::CAM_DET => &[Sensor::Camera],
        StreamId::LIDAR | StreamId::LIDAR_DET | StreamId::LIDAR_OBJ => &[Sensor::Lidar],
        // The fusion's output and the answer: an instant either sensor's
        // source lacks makes no set.
        StreamId::TRACKS | StreamId::EGO => &[Sensor::Camera, Sensor::Lidar],
        _ => &[],
    }
}

/// Previous *delivered* `seq` per edge.
#[derive(Default)]
struct SeqGap(BTreeMap<&'static str, u64>);

impl SeqGap {
    /// Records one delivered `seq` and returns how many frames of sequence
    /// space it advanced past the previous delivery on that edge: 1 when the
    /// consumer kept up, 3 when it never saw two frames in between.
    ///
    /// `None` for the first delivery on an edge, which has no predecessor to
    /// measure against. That row is skipped rather than logged as 1, which
    /// would claim "nothing was missed" on evidence we do not have, or as 0,
    /// which reads as a stall.
    ///
    /// The gap is computed in `f64` rather than `u64` so that a seq that did
    /// not advance shows up as a visible zero or negative rather than
    /// wrapping into a huge number.
    fn observe(&mut self, edge: &'static str, seq: u64) -> Option<f64> {
        self.0
            .insert(edge, seq)
            .map(|prev| seq as f64 - prev as f64)
    }
}

/// What the run's arguments and its drive say about how to draw it: the
/// camera picture's size, the sensor period the pairing plot is scaled by,
/// which camera queue feeds the answer and what slows it, and which lanes
/// and pictures the run can mark. Computed in `run.rs` from the run's own
/// arguments — the dashboard only draws from them and never measures
/// anything itself.
#[derive(Clone, Debug, Default)]
pub struct Bounds {
    /// The camera frame's width and height in pixels, from frame 0's PNG
    /// header: the camera view's bounds, so the picture fills its view edge
    /// to edge from the first frame instead of fitting whatever was drawn
    /// first. `None` when the header could not be read, and the view then
    /// fits the image the viewer's own way.
    pub image_wh: Option<(f32, f32)>,
    /// One sensor period in ms on the SENSOR clock: the camera's median
    /// frame period, which on these drives is the lidar's sweep period too
    /// (both run at 10 Hz). The axes of the plots drawn in sensor time -- the
    /// pairing window -- are scaled by it. The driver's documented default
    /// period stands in on a drive with fewer than two frames; `None` only
    /// in a `Bounds` built without a drive.
    pub period_ms: Option<f64>,
    /// The camera queue whose output reaches the answer, where the camera
    /// knobs act (`run::camera_queues`): `cam0->camdet` with the detector,
    /// `cam0->proc` without. The one the `camera_queue` lane watches, and
    /// the camera stage the pairing plot's camera dot and the stage table's
    /// camera row read.
    pub camera_edge: Option<&'static str>,
    /// `--consumer-delay-ms`: the sleep inside that camera stage's measured
    /// window. The stage table says so beside the stage's time, which it is
    /// part of.
    pub camera_delay_ms: u64,
    /// `--rate`, when it is finite: how many sensor milliseconds one host
    /// millisecond is, for the ready dots, which are timed on the host clock
    /// and drawn on the sensor's. `None` on an unpaced run, which has no
    /// deadlines to time them from.
    pub rate: Option<f64>,
    /// Whether the run has `velo->reduce`, the edge the lidar lane watches.
    /// Only then does a frame absent in the source mark that lane: on a run
    /// without it no row would ever step the lane back down, and `no sweep`
    /// would stand to the end of the recording.
    pub lidar_lane: bool,
    /// Whether the chain ends in an answer (`track` runs). Only then does a
    /// frame absent in the source put its own headline up and mark the
    /// pairing lane, for the same reason: a run with no answer has nothing
    /// to replace them afterwards.
    pub answer: bool,
}

/// The live dashboard (M8): mirrors evidence rows onto the run's Rerun
/// recording on the same three timelines the image uses, so a frame and its
/// timing scrub together -- the health lanes, the pairing's ready dots, the
/// pipeline graph and the two tables. Constructed only with `--dashboard`.
///
/// No legend document travels with the recording: the entity names and the
/// view titles have to carry their own meaning, and a reader who needs more
/// has the README ("The dashboard") rather than a panel in the viewer.
pub struct Dashboard {
    rec: Canvas,
    /// The sample of the last row drawn. An event is drawn at that row's
    /// timelines but `host` ([`Dashboard::log_event`]), and so are the tables
    /// at the end, so their drawings are filed under it.
    last: Subject,
    /// Whether [`Dashboard::finish`] has run.
    finished: bool,
    gaps: SeqGap,
    bounds: Bounds,
    /// Entity paths that already carry their static `SeriesPoints` styling,
    /// registered on each path's first scalar.
    styled: BTreeSet<String>,
    /// Lane entities that already carry their static `StateConfiguration`.
    lanes: BTreeSet<String>,
    /// Each sticky lane's state, by lane name ([`crate::dashboard::LANE_NAMES`]).
    sticky: BTreeMap<&'static str, Sticky>,
    /// The latest delivered `payload_bytes` per edge and per producer, for
    /// the graph's labels: the graph is one picture per answer, and the rows
    /// that feed it arrive one edge at a time.
    bytes: BTreeMap<&'static str, u64>,
    /// Every delivered row on the tables' edges ([`TABLE_EDGES`]), for their
    /// medians.
    table: BTreeMap<&'static str, EdgeRows>,
    /// Each sweep's two halves until both are in, by frame number.
    halves: BTreeMap<u64, Halves>,
    /// Sweeps whose camera and lidar finishes were both seen, and of them
    /// those whose camera finished after the lidar's.
    camera_later: (u64, u64),
    /// Answers seen, so the tables are first written after
    /// [`TABLE_AFTER_ANSWERS`].
    answers: u64,
    /// Whether the pipeline graph's edges are logged yet.
    graphed: bool,
    /// Each producer row's `storage_id`, by `(stream, seq)`, so a consumer's
    /// row can say whether it read that buffer or a copy. Grows with the run
    /// at 16 bytes a sample, like the rows the `rec` thread already keeps.
    produced: BTreeMap<(u8, u64), usize>,
    /// Frames the source has no sample for, by `(sensor, frame)`, so a
    /// queue's skip over them -- the lidar edges go from sweep 176 to 181 on
    /// drive 0009 -- is not drawn as the queue's own loss, on the sensor's
    /// own edges or on any edge of a stream that carries its frame numbers
    /// ([`sensors_of`]).
    absent: BTreeSet<(Sensor, u64)>,
}

impl Dashboard {
    /// Sends the layout first, before any row and before the clock starts,
    /// straight to the recording: a viewer the run spawned opens straight
    /// into it, and a file carries it at its head. Every row after it is
    /// drawn on `canvas`.
    pub fn new(
        rec: &RecordingStream,
        canvas: Canvas,
        bounds: Bounds,
        mode: Mode,
    ) -> Result<Self, rerun::RecordingStreamError> {
        send_blueprint(rec, mode, &bounds)?;
        Ok(Dashboard {
            rec: canvas,
            last: Subject::default(),
            finished: false,
            gaps: SeqGap::default(),
            bounds,
            styled: BTreeSet::new(),
            lanes: BTreeSet::new(),
            sticky: BTreeMap::new(),
            bytes: BTreeMap::new(),
            table: BTreeMap::new(),
            halves: BTreeMap::new(),
            camera_later: (0, 0),
            answers: 0,
            graphed: false,
            produced: BTreeMap::new(),
            absent: BTreeSet::new(),
        })
    }

    /// A frame a sensor's source has no sample for (`absent_in_source`), at
    /// the slot's own place on `host`, where the replay reported it.
    ///
    /// The sensor's own lane turns grey -- `no sweep` on the lidar's, `no
    /// frame` on the camera's -- and its pictures of the sample before are
    /// cleared, so an old cloud or frame does not stand in for an instant it
    /// does not show. For a missing sweep, with an answer, the pairing lane
    /// turns grey `source gap` too and the answer's own pictures and headline
    /// say there is none for this frame, and why ([`log_no_sweep`]); for a
    /// missing camera frame the sweep of that number reaches the fusion and
    /// expires as `pair_absent_in_source`, which marks the pairing lane and
    /// draws the headline itself. The frame's WARN row, and the run's one
    /// `SourceGap` event naming the whole gap, are logged where every drop and
    /// event is.
    fn log_absent(&mut self, e: &Evidence) -> Result<(), RecError> {
        let sensor = if e.edge == "cam0" {
            Sensor::Camera
        } else {
            Sensor::Lidar
        };
        self.absent.insert((sensor, e.seq));
        if e.edge == "cam0" {
            self.sticky(LANE_CAMERA_NAME, Some((NO_FRAME, 2)))?;
            log_no_frame(&self.rec, e.seq, self.bounds.answer)?;
            return Ok(());
        }
        if self.bounds.lidar_lane {
            self.sticky(LANE_LIDAR_NAME, Some((NO_SWEEP, 2)))?;
        }
        for path in [
            entity::LIDAR_SWEEP,
            entity::LIDAR_VOXELS,
            entity::LIDAR_DETECTIONS,
            entity::LIDAR_TRACKS,
        ] {
            self.rec.log(path, &Clear::flat())?;
        }
        if self.bounds.answer {
            self.sticky(LANE_PAIRING_NAME, Some((SOURCE_GAP, 2)))?;
            log_no_sweep(&self.rec, e.seq)?;
        }
        Ok(())
    }

    /// How many frames absent in the source a delivery on `e`'s stream
    /// stepped over since the one before it, `gap` sequence numbers back: the
    /// frames in between that the source of any sensor the stream's `seq`
    /// counts never had. By sensor, not by stream: `cam_det->track` carries
    /// the camera's frame numbers on its own stream id, and a camera frame
    /// absent in the source is as absent there as on `cam0`'s own edges.
    fn absent_between(&self, e: &Evidence, gap: f64) -> f64 {
        if gap <= 1.0 {
            return 0.0;
        }
        let prev = e.seq.saturating_sub(gap as u64);
        let frames: BTreeSet<u64> = sensors_of(e.stream)
            .iter()
            .flat_map(|&s| {
                self.absent
                    .range((s, prev + 1)..(s, e.seq))
                    .map(|&(_, f)| f)
            })
            .collect();
        frames.len() as f64
    }

    /// Logs one state onto a lane, configuring the lane's states and colours
    /// the first time it is seen.
    fn lane(&mut self, path: &str, state: &str) -> Result<(), RecError> {
        if !self.lanes.contains(path) {
            if let Some(lane) = lane_style(path) {
                self.rec.log_static(path.to_string(), &lane.archetype())?;
            }
            self.lanes.insert(path.to_string());
        }
        self.rec
            .log(path.to_string(), &StateChange::single(state))?;
        Ok(())
    }

    /// Feeds one row's event to the sticky lane `name`, and logs the lane's
    /// state only when it changed (or on its first row).
    fn sticky(&mut self, name: &'static str, event: LaneEvent) -> Result<(), RecError> {
        let camera_edge = self.bounds.camera_edge;
        let changed = self
            .sticky
            .entry(name)
            .or_insert_with(|| Sticky::new(lane_clear_after(name, camera_edge)))
            .observe(event);
        if let Some(word) = changed {
            self.lane(&lane_path(name), word)?;
        }
        Ok(())
    }

    /// Logs one scalar at `<root>/<leaf>`, naming, colouring and marking that
    /// entity the first time it is seen.
    fn scalar(&mut self, root: &str, leaf: &str, v: f64) -> Result<(), RecError> {
        let path = format!("{root}/{leaf}");
        if !self.styled.contains(&path) {
            if let Some(style) = series_style(leaf) {
                self.rec
                    .log_static(path.clone(), style.archetype().as_ref())?;
            }
            self.styled.insert(path.clone());
        }
        self.rec.log(path, &Scalars::single(v))?;
        Ok(())
    }

    /// One ready dot: on the zoom onto the sweep's end ([`zoom_ms`]), and,
    /// unless it is the zoom's alone, on the pairing plot ([`done_ms`]).
    /// None on an unpaced run, which has no deadline to time it from.
    fn done_dot(
        &mut self,
        dot: &ReadyDot,
        span_ns: i64,
        after_due_ns: i64,
    ) -> Result<(), RecError> {
        let (Some(rate), Some(period)) = (self.bounds.rate, self.bounds.period_ms) else {
            return Ok(());
        };
        if dot.overview {
            self.scalar(
                entity::PAIRING,
                dot.leaf,
                done_ms(span_ns, after_due_ns, rate, period),
            )?;
        }
        self.scalar(
            entity::AFTER_SWEEP,
            dot.leaf,
            zoom_ms(after_due_ns, rate, period),
        )
    }

    /// A delivered row's ready dot ([`READY_DOTS`]): a lidar stage's and the
    /// answer's from their own rows, which carry their sweep's due and range,
    /// the moment the row arrives; the camera stage's once its sweep's due
    /// is in ([`Halves`]) -- which also counts, once `detect`'s finish is in
    /// too, whether the camera finished after the lidar. The sweep's own row
    /// draws the zoom's line at the sweep's end.
    fn log_done(&mut self, e: &Evidence) -> Result<(), RecError> {
        let span = e.tov_end_ns - e.tov_start_ns;
        if let (Some(due), Some(end)) = (e.due_ns, e.proc_end_ns) {
            if let Some(dot) = READY_DOTS
                .iter()
                .find(|d| d.clock == StageClock::Edge(e.edge))
            {
                self.done_dot(dot, span, end - due)?;
            }
        }
        if e.edge == "velo" && e.due_ns.is_some() && self.bounds.rate.is_some() {
            self.scalar(entity::AFTER_SWEEP, SWEEP_END_LEAF, 0.0)?;
        }
        let (sweep, lidar, camera) = match e.edge {
            "velo" => (e.due_ns.map(|d| (d, span)), None, None),
            "det->detect" => (None, e.proc_end_ns, None),
            edge if self.bounds.camera_edge == Some(edge) => (None, None, e.proc_end_ns),
            _ => return Ok(()),
        };
        let h = self.halves.entry(e.seq).or_default();
        h.sweep = h.sweep.or(sweep);
        h.lidar = h.lidar.or(lidar);
        h.camera = h.camera.or(camera);
        let dot = match (h.sweep, h.camera) {
            (Some((due, span)), Some(c)) if !h.drawn => {
                h.drawn = true;
                Some((span, c - due))
            }
            _ => None,
        };
        let later = match (h.camera, h.lidar) {
            (Some(c), Some(l)) if !h.compared => {
                h.compared = true;
                Some(c > l)
            }
            _ => None,
        };
        if h.drawn && h.compared {
            self.halves.remove(&e.seq);
        }
        let newest = e.seq;
        self.halves.retain(|&seq, _| seq + HALVES_KEPT >= newest);
        if let Some(later) = later {
            self.camera_later.0 += 1;
            self.camera_later.1 += u64::from(later);
        }
        if let (Some((span, after_due)), Some(camera_dot)) = (
            dot,
            READY_DOTS.iter().find(|d| d.clock == StageClock::Camera),
        ) {
            self.done_dot(camera_dot, span, after_due)?;
        }
        Ok(())
    }

    /// Files a delivered row on one of the tables' edges into their
    /// medians, and, on a queue's row, whether its consumer read the buffer
    /// its producer handed on. The producer's row precedes the consumer's on
    /// the evidence channel in practice (it is sent at admission, the
    /// consumer's after `proc_end`); when it does not, that row is counted
    /// neither way.
    fn observe_tables(&mut self, e: &Evidence) {
        if !TABLE_EDGES.contains(&e.edge) {
            return;
        }
        let rows = self.table.entry(e.edge).or_default();
        rows.carried.push(e.payload_bytes);
        rows.alloc.push(e.bytes_alloc);
        if e.edge.contains("->") {
            if let Some(t) = e.proc_start_ns.zip(e.proc_end_ns).map(|(s, t)| t - s) {
                rows.time_ns.push(t);
            }
            if self
                .produced
                .get(&(e.stream, e.seq))
                .is_some_and(|&p| p != e.storage_id)
            {
                rows.copied += 1;
            }
        } else if ADMISSION_EDGES.contains(&e.edge) {
            rows.time_ns.push(e.decode_ns);
        }
    }

    /// The pipeline graph for the answer that just arrived: its edges once,
    /// statically, then its nodes at this answer's timelines, each labelled
    /// with what its stage handed on for the latest sweep and coloured by
    /// the sensor or family it belongs to. The three nodes the byte chain
    /// does not describe -- the camera, the admission and the camera stage --
    /// carry a second line saying what they do.
    fn log_graph(&mut self, e: &Evidence) -> Result<(), RecError> {
        if !self.graphed {
            self.rec.log_static(
                entity::GRAPH_PIPELINE,
                &GraphEdges::new(GRAPH_EDGES).with_directed_edges(),
            )?;
            self.graphed = true;
        }
        let carried = |edge: &str| self.bytes.get(edge).map(|b| human_bytes(*b));
        let camera_stage = self
            .bounds
            .camera_edge
            .and_then(|e| e.split_once("->"))
            .map_or("camera", |(_, stage)| stage);
        let label = |id: &str| -> String {
            let with = |what: &str, edge: &str| match carried(edge) {
                Some(b) => format!("{what} {b}"),
                None => what.to_string(),
            };
            match id {
                "cam0" => format!("{}\ndecoded from PNG", with("camera", "cam0")),
                "velo" => with("lidar", "velo"),
                "admit" => "admission\none arrival order".to_string(),
                "camera" => format!(
                    "{}\n{}",
                    with(camera_stage, "cam_det"),
                    camera_words(camera_stage)
                ),
                "reduce" => with("voxels", "det"),
                "detect" => with("dets", "obj"),
                "track" => with("tracks", "track"),
                "state" => with("state", "state"),
                _ => match e.measurement_age_ns {
                    Some(age) => format!("answer {:.0} ms", age as f64 * 1e-6),
                    None => "answer".to_string(),
                },
            }
        };
        let rgb = |id: &str| match id {
            "cam0" | "camera" => C_CAMERA_ONLY,
            "velo" | "reduce" => C_LIDAR_ONLY,
            "detect" | "track" | "state" => C_DETECTION,
            "answer" => C_ANSWER,
            _ => C_BOUND,
        };
        let nodes = GraphNodes::new(GRAPH_NODES.map(|(id, _, _)| id))
            .with_positions(GRAPH_NODES.map(|(_, x, y)| [x, y]))
            .with_labels(GRAPH_NODES.map(|(id, _, _)| label(id)))
            .with_colors(GRAPH_NODES.map(|(id, _, _)| {
                let c = rgb(id);
                Color::from_rgb(c[0], c[1], c[2])
            }))
            .with_radii([GRAPH_NODE_RADIUS; GRAPH_NODES.len()])
            .with_show_labels(true);
        self.rec.log(entity::GRAPH_PIPELINE, &nodes)?;
        Ok(())
    }

    /// The two tables, from the medians of every delivered row so far,
    /// logged statically so the last write is the one a reader sees: the
    /// byte chain once the run has answered, since it ends at the answer,
    /// and the stages whenever a stage has run.
    fn log_tables(&self) -> Result<(), RecError> {
        if self.answers > 0 {
            self.log_byte_table()?;
        }
        self.log_stage_table()
    }

    /// The byte table ([`byte_table`]), a line per link of [`CHAIN_BYTES`].
    fn log_byte_table(&self) -> Result<(), RecError> {
        let rows = |edge: &str| self.table.get(edge);
        let links: Vec<ByteLink> = CHAIN_BYTES
            .iter()
            .map(|l| ByteLink {
                name: l.name,
                step: l.step,
                carried: rows(l.edge).and_then(|r| median(&r.carried)),
                built: rows(l.built_by).and_then(|r| median(&r.alloc)),
                read: rows(l.edge).and_then(|r| median(&r.alloc)),
                rows: rows(l.edge).map_or(0, |r| r.carried.len()),
                copied: rows(l.edge).map_or(0, |r| r.copied),
            })
            .collect();
        let sweeps = rows("velo").map_or(0, |r| r.carried.len());
        self.rec.log_static(
            entity::BYTES_TABLE,
            &TextDocument::new(byte_table(&links, sweeps)).with_media_type(MediaType::markdown()),
        )?;
        Ok(())
    }

    /// The stage table ([`stage_table`]): every stage the run has rows for,
    /// and what their times are and are not.
    fn log_stage_table(&self) -> Result<(), RecError> {
        let rows = |edge: &str| self.table.get(edge);
        let stages = stage_rows(self.bounds.camera_edge);
        let lines: Vec<StageLine> = stages
            .iter()
            .filter(|s| rows(s.timed_on).is_some() || rows(s.hands_on).is_some())
            .map(|s| StageLine {
                row: s,
                time_ns: rows(s.timed_on).and_then(|r| median(&r.time_ns)),
                carried: rows(s.hands_on).and_then(|r| median(&r.carried)),
            })
            .collect();
        if lines.is_empty() {
            return Ok(());
        }
        // The camera stage is the second row, whichever it is.
        let notes = self.stage_notes(&stages[1]);
        self.rec.log_static(
            entity::STAGE_TABLE,
            &TextDocument::new(stage_table(&lines, &notes)).with_media_type(MediaType::markdown()),
        )?;
        Ok(())
    }

    /// What the stage table's times are, and what they are not: the times
    /// themselves; the camera stage's sleep, its working memory and the
    /// frames it did not run on; and how often its result came after the
    /// lidar's.
    fn stage_notes(&self, camera: &StageRow) -> Vec<String> {
        let rows = |edge: &str| self.table.get(edge);
        let stage = camera.stage;
        let mut notes = vec![
            "A time is the stage's median per sample: its own work, from `proc_start` to \
             `proc_end`, or a sensor driver's decode. track's includes any wait for the camera."
                .to_string(),
        ];
        if self.bounds.camera_delay_ms > 0 {
            notes.push(format!(
                "{stage}'s includes the {} ms sleep of `--consumer-delay-ms`.",
                self.bounds.camera_delay_ms
            ));
        }
        if let Some(r) = rows(camera.timed_on) {
            let out = rows(camera.hands_on).and_then(|h| median(&h.carried));
            if let (Some(alloc), Some(out)) = (median(&r.alloc), out) {
                if alloc > 0 {
                    let why = if stage == "camdet" {
                        ", the network's working memory"
                    } else {
                        ""
                    };
                    notes.push(format!(
                        "{stage} allocates {} while it works on each frame{why}, and hands on {}.",
                        human_bytes(alloc),
                        human_bytes(out)
                    ));
                }
            }
            let frames = rows("cam0").map_or(0, |c| c.carried.len());
            if r.carried.len() < frames {
                notes.push(format!(
                    "{stage} ran on {} of the {frames} frames the camera delivered.",
                    r.carried.len()
                ));
            }
        }
        match self.camera_later {
            (0, _) => {}
            (n, 0) => notes.push(format!(
                "The camera's result was ready before the lidar's boxes on all {n} sweeps."
            )),
            (n, later) => notes.push(format!(
                "The camera's result was ready after the lidar's boxes on {later} of {n} sweeps: \
                 on the Demo page, the blue diamond above `detect`'s teal one."
            )),
        }
        notes
    }

    /// At the end of the run, once every stage has stopped: the tables from
    /// the whole run's medians. The dashboard draws last, so it then closes
    /// the viewer's queue, and the viewer thread drains it and stops. Runs
    /// once.
    fn finish(&mut self) -> Result<(), RecError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let drawn = self.log_tables();
        self.rec.send(self.last.clone());
        self.rec.close();
        drawn
    }

    /// One evidence row, drawn as one drawing of the row's sample.
    ///
    /// Not the viewer's own rows (`draw->viewer`): drawn, each would be a
    /// drawing, and each drawing another row.
    fn log_row(&mut self, e: &Evidence) -> Result<(), RecError> {
        if e.edge == VIEWER_EDGE {
            return Ok(());
        }
        let drawn = self.draw_row(e);
        self.last = Subject::of_row(e);
        self.rec.send(self.last.clone());
        drawn
    }

    /// Stamps the three timelines from the row itself, then draws what the
    /// row says: a queue's row on its lane and the fusion's on the pairing
    /// lane, a frame the source never had on its sensor's lane, a half's or
    /// the answer's finish as a ready dot on the pairing plot, every row on
    /// the tables' edges into their medians, the answer on the graph, and
    /// every loss as a WARN row.
    fn draw_row(&mut self, e: &Evidence) -> Result<(), RecError> {
        // A row with no time of validity -- a frame absent in the source,
        // whose `tov_*_ns` are 0 because the source measured nothing -- has
        // no place on the sensor clock, and stamped at 0 it would put a point
        // in 1970 and stretch that timeline over forty years. It is logged on
        // `host` and `seq` only.
        if e.tov_start_ns == 0 && e.tov_end_ns == 0 {
            self.rec.disable_timeline("sensor_time");
        } else {
            self.rec
                .set_timestamp_nanos_since_epoch("sensor_time", e.tov_start_ns);
        }
        self.rec
            .set_duration_secs("host", e.arrival_ns as f64 * 1e-9);
        self.rec.set_time_sequence("seq", e.seq as i64);

        let real_queue = e.edge.contains("->");
        let delivered = e.outcome == Outcome::Delivered;

        if real_queue {
            // What the queue did with this sample, on its sticky lane: the
            // gap is measured only between deliveries on a real queue -- the
            // driver's `cam0` pseudo-edge has no consumer behind it to fall
            // behind. Frames the source never had are not the queue's to
            // lose: a lidar edge's 176 -> 181 on drive 0009 is one step.
            let gap = delivered
                .then(|| self.gaps.observe(e.edge, e.seq))
                .flatten()
                .map(|g| g - self.absent_between(e, g));
            if let Some(lane) = queue_lane(e.edge, self.bounds.camera_edge) {
                self.sticky(lane, queue_event(e.outcome, gap))?;
            }
        } else if delivered {
            // A producer's row: remember which buffer it handed on, so each
            // consumer's row can say whether it read that one.
            self.produced.insert((e.stream, e.seq), e.storage_id);
        }
        // The fusion's own row (`track`, the produced or the missing one),
        // on the pairing lane: `reason` is blank on a clean pair and names
        // the failure otherwise, which no other row carries.
        if e.edge == "track" {
            self.sticky(LANE_PAIRING_NAME, pairing_event(e))?;
        }
        // A frame a sensor's source never had: that driver's own row, the
        // only one there is for it anywhere in the pipeline.
        if !delivered && ADMISSION_EDGES.contains(&e.edge) && e.reason == ABSENT_IN_SOURCE {
            self.log_absent(e)?;
        }

        if delivered {
            self.log_done(e)?;
            self.observe_tables(e);
            self.bytes.insert(e.edge, e.payload_bytes);
        }
        if delivered && e.edge == "state->sink" {
            // The answer arrived: the pipeline graph, labelled with this
            // sweep's bytes, and, after the first ten answers, the tables.
            self.log_graph(e)?;
            self.answers += 1;
            if self.answers == TABLE_AFTER_ANSWERS {
                self.log_tables()?;
            }
        }

        if !delivered {
            // One entity for every edge, and the only place a drop's REASON
            // appears. The edge is in the text because the entity does not
            // say it.
            self.rec.log(
                entity::LOG_DROPS,
                &TextLog::new(format!(
                    "{} seq {} {:?}{}{}",
                    e.edge,
                    e.seq,
                    e.outcome,
                    if e.reason.is_empty() { "" } else { " " },
                    e.reason
                ))
                .with_level(TextLogLevel::WARN),
            )?;
        }
        Ok(())
    }

    /// A rare run-level event (D15) on `log/events`: a clean shutdown is
    /// information, a gap in a sensor's source a warning, a degraded recorder
    /// an error.
    ///
    /// Only `host` is stamped from the event. `seq` is left where the last
    /// evidence row put it -- a per-stream frame number -- because stamping
    /// the event's `arrival_seq_after` there once stretched the `seq` axis of
    /// every plot to the admission count of the whole run. For the same
    /// reason its drawing is filed under that row's sample.
    fn log_event(&self, ev: &Event) -> Result<(), RecError> {
        let rec = &self.rec;
        rec.set_duration_secs("host", ev.host_ns as f64 * 1e-9);
        let level = match ev.kind {
            EventKind::Shutdown => TextLogLevel::INFO,
            EventKind::SourceGap => TextLogLevel::WARN,
            EventKind::RecorderDegraded => TextLogLevel::ERROR,
        };
        let drawn = rec.log(
            entity::LOG_EVENTS,
            &TextLog::new(format!("{:?} at {}: {}", ev.kind, ev.stage, ev.detail))
                .with_level(level),
        );
        rec.send(self.last.clone());
        Ok(drawn?)
    }
}

impl Drop for Dashboard {
    /// However the `rec` thread ends -- an error, a panic -- the viewer's
    /// queue is closed, so the viewer thread is never left waiting on a
    /// producer that is gone.
    fn drop(&mut self) {
        self.rec.close();
    }
}

/// How many evidence rows the `rec` thread buffers before forcing a flush, so
/// an abnormal exit loses a bounded tail rather than everything buffered.
const FLUSH_EVERY: usize = 64;

/// Column names of `events.csv`, in `Event` field order (pinned by a test).
pub const EVENT_HEADER: [&str; 6] = [
    "run_id",
    "host_ns",
    "arrival_seq_after",
    "kind",
    "stage",
    "detail",
];

/// Errors of the `rec` thread.
#[derive(Debug)]
pub enum RecError {
    Csv(csv::Error),
    Io(std::io::Error),
    Rerun(rerun::RecordingStreamError),
}

impl std::fmt::Display for RecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecError::Csv(e) => write!(f, "csv: {e}"),
            RecError::Io(e) => write!(f, "io: {e}"),
            RecError::Rerun(e) => write!(f, "dashboard: {e}"),
        }
    }
}

impl std::error::Error for RecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RecError::Csv(e) => Some(e),
            RecError::Io(e) => Some(e),
            RecError::Rerun(e) => Some(e),
        }
    }
}

impl From<csv::Error> for RecError {
    fn from(e: csv::Error) -> Self {
        RecError::Csv(e)
    }
}

impl From<std::io::Error> for RecError {
    fn from(e: std::io::Error) -> Self {
        RecError::Io(e)
    }
}

impl From<rerun::RecordingStreamError> for RecError {
    fn from(e: rerun::RecordingStreamError) -> Self {
        RecError::Rerun(e)
    }
}

/// Body of the `rec` thread: drains `q` into `<dir>/evidence.csv` and
/// `<dir>/events.csv` (headers written on the first row of each) and
/// returns the evidence rows for the summary, so nothing reads CSV back.
/// With `Some(dashboard)` each row is also logged to Rerun (M8).
pub fn rec_thread(
    q: Arc<BoundedQueue<EvRow>>,
    dir: PathBuf,
    mut dashboard: Option<Dashboard>,
) -> Result<Vec<Evidence>, RecError> {
    set_stage_slot(SLOT_REC);
    let mut w = csv::Writer::from_path(dir.join("evidence.csv"))?;
    // Events are rare and csv writes headers lazily, so an event-free run
    // would leave a 0-byte file: write the header eagerly instead.
    let mut ew = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(dir.join("events.csv"))?;
    ew.write_record(EVENT_HEADER)?;
    ew.flush()?;
    let mut rows = Vec::new();
    // Flush every `FLUSH_EVERY` rows rather than only at the end: if the
    // process is killed (Ctrl-C, SIGKILL, a panic that tears the process down
    // before this thread is joined), the `BufWriter` inside `csv::Writer` is
    // destroyed without running, and everything still buffered is lost —
    // truncating `evidence.csv` mid-row. Periodic flushing bounds that loss to
    // the last `FLUSH_EVERY` rows instead of "everything since the last 8 KiB
    // boundary", and a row is only ever written whole.
    let mut since_flush = 0usize;
    while let Some(env) = q.pop() {
        match env.item {
            EvRow::Evidence(e) => {
                w.serialize(&e)?;
                if let Some(d) = dashboard.as_mut() {
                    d.log_row(&e)?;
                }
                rows.push(e);
                since_flush += 1;
                if since_flush >= FLUSH_EVERY {
                    w.flush()?;
                    since_flush = 0;
                }
            }
            EvRow::Event(ev) => {
                ew.serialize(&ev)?;
                // Events are rare and always interesting; never leave one buffered.
                ew.flush()?;
                if let Some(d) = dashboard.as_mut() {
                    d.log_event(&ev)?;
                    // The shutdown event is sent once every stage has
                    // stopped, so nothing the dashboard draws comes after
                    // it: it finishes here, and closes the viewer's queue,
                    // while the viewer's own rows are still to come.
                    if matches!(ev.kind, EventKind::Shutdown) {
                        d.finish()?;
                    }
                }
            }
        }
    }
    if let Some(d) = dashboard.as_mut() {
        d.finish()?;
    }
    w.flush()?;
    ew.flush()?;
    Ok(rows)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use pipes_core::clock::HostTime;
    use pipes_core::evidence::RowCtx;

    use super::*;
    use crate::viewer::ViewerQueue;

    #[test]
    fn event_header_matches_the_struct() {
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        let ev = Event::recorder_degraded(&ctx, HostTime(1), "proc", 2);
        let mut w = csv::Writer::from_writer(Vec::new());
        w.serialize(&ev).unwrap();
        let text = String::from_utf8(w.into_inner().unwrap()).unwrap();
        assert_eq!(text.lines().next().unwrap(), EVENT_HEADER.join(","));
    }

    #[test]
    fn rec_thread_writes_an_events_header_even_without_events() {
        let dir = std::env::temp_dir().join(format!("pipes-rec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let q: Arc<BoundedQueue<EvRow>> = Arc::new(BoundedQueue::new(4, QueuePolicy::DropOldest));
        q.close();
        let rows = rec_thread(Arc::clone(&q), dir.clone(), None).unwrap();
        assert!(rows.is_empty());
        let events = std::fs::read_to_string(dir.join("events.csv")).unwrap();
        assert_eq!(events.trim_end(), EVENT_HEADER.join(","));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_logged_series_has_a_name_and_a_colour() {
        for leaf in SERIES_LEAVES {
            let style = series_style(leaf)
                .unwrap_or_else(|| panic!("{leaf} is logged but has no SeriesLines style"));
            assert!(!style.label.is_empty(), "{leaf} has an empty legend label");
            assert!(style.width > 0.0, "{leaf} has a zero stroke width");
        }
        assert!(series_style("not_a_series").is_none());
    }

    #[test]
    fn name_fits_a_legend_and_carries_its_unit() {
        // A legend entry is read at a glance beside the plot, so it is short,
        // and it says what the number IS, so it ends in its unit in
        // parentheses.
        for leaf in SERIES_LEAVES {
            let label = series_style(leaf).unwrap().label;
            assert!(
                label.chars().count() <= LABEL_MAX,
                "{leaf}: {label:?} is longer than {LABEL_MAX} characters"
            );
            assert!(
                label.ends_with(')') && label.contains(" ("),
                "{leaf}: {label:?} does not end in a unit"
            );
        }
    }

    #[test]
    fn the_time_to_contact_is_points_in_the_warning_red_against_a_grey_line() {
        // The flagged object changes from sweep to sweep, so the TTC is drawn
        // as markers: a line joined one object's approach to the next one's
        // with a jump no object made.
        let ttc = series_style("ttc_s").unwrap();
        assert_eq!(ttc.mark, Mark::Points(MarkerShape::Circle));
        // Red: the one demo number that is bad when it is small. Never the
        // answer's gold, which is the box and its words.
        assert!(
            ttc.rgb[0] > 200 && ttc.rgb[0] > 2 * ttc.rgb[1] && ttc.rgb[0] > 2 * ttc.rgb[2],
            "time to contact is not in the warning family: {:?}",
            ttc.rgb
        );
        assert_ne!(ttc.rgb, C_ANSWER);
        // The 3 s line it is read against is a reference: a grey line.
        let warn = series_style("ttc_warn").unwrap();
        assert!(matches!(warn.mark, Mark::Line { .. }));
        assert!(warn.rgb[0] == warn.rgb[1] && warn.rgb[1] == warn.rgb[2]);
        // And its archetype is the points one, not a line with a marker.
        let arch = series_archetype("ttc_s").unwrap();
        let names: Vec<String> = arch
            .as_serialized_batches()
            .iter()
            .map(|b| b.descriptor.component.to_string())
            .collect();
        assert!(
            names.iter().any(|n| n.contains("SeriesPoints")),
            "{names:?}"
        );
    }

    #[test]
    fn a_stage_that_logs_its_own_series_gets_the_same_styling_table() {
        // `series_archetype` is the door every stage that logs a series comes
        // through, because an evidence row carries how long a sample took and
        // never what it SAID. It must answer for exactly the leaves
        // `series_style` does, or the test above is checking a table nothing
        // reads.
        for leaf in SERIES_LEAVES {
            assert!(
                series_archetype(leaf).is_some(),
                "{leaf} is styled for the dashboard but not for a stage that logs it itself"
            );
        }
        assert!(series_archetype("not_a_series").is_none());
    }

    #[test]
    fn a_reference_line_is_grey_flat_and_never_heavier_than_what_it_bounds() {
        // A reference line sits behind the measurement it bounds rather than
        // competing with it: the one bright grey, a step, and no heavier.
        for (reference, measured) in [
            ("ttc_warn", "ttc_s"),
            ("sweep_start", "camera"),
            ("sweep_end", "camera"),
            ("trigger", "camera"),
        ] {
            let r = series_style(reference).unwrap();
            let m = series_style(measured).unwrap();
            assert_eq!(r.rgb, C_REFERENCE, "{reference}");
            assert!(r.step(), "{reference} is not flat");
            assert!(r.width <= m.width, "{reference} is heavier than {measured}");
        }
    }

    #[test]
    fn the_ready_dots_are_diamonds_in_the_colour_of_their_half() {
        // A diamond is a stage finishing, beside the circle of the camera
        // frame's instant, in the colour of what it is about.
        let marks: Vec<(&str, [u8; 3], Mark)> = READY_DOTS
            .iter()
            .map(|d| {
                let s = series_style(d.leaf).unwrap();
                (d.leaf, s.rgb, s.mark)
            })
            .collect();
        let diamond = Mark::Points(MarkerShape::Diamond);
        assert_eq!(
            marks,
            [
                ("1_camera_done", C_CAMERA_ONLY, diamond),
                ("2_reduce_done", C_LIDAR_EARLY, diamond),
                ("3_detect_done", C_LIDAR_ONLY, diamond),
                ("4_answer_done", C_ANSWER, diamond),
            ]
        );
        // `reduce`'s is the lidar's teal, lighter: brighter on every channel.
        assert!(C_LIDAR_EARLY.iter().zip(C_LIDAR_ONLY).all(|(a, b)| *a > b));
        assert_eq!(
            series_style("camera").unwrap().mark,
            Mark::Points(MarkerShape::Circle)
        );
        // The two shares are the colours of the sensors whose objects they
        // are shares of.
        assert_eq!(series_style("fused_fraction").unwrap().rgb, C_LIDAR_ONLY);
        assert_eq!(
            series_style("camera_fused_fraction").unwrap().rgb,
            C_CAMERA_ONLY
        );
    }

    #[test]
    fn a_ready_dot_is_drawn_in_ms_from_its_sweep_s_start() {
        const MS: i64 = 1_000_000;
        let p = 103.3;
        let sweep = 103_300_000;
        // The lidar's boxes 9.1 ms after the sweep ended; the camera's
        // result 4.6 ms before it.
        assert!((done_ms(sweep, 9_100_000, 1.0, p) - 112.4).abs() < 1e-9);
        assert!((done_ms(sweep, -4_600_000, 1.0, p) - 98.7).abs() < 1e-9);
        // At twice real time a host millisecond is two of the sensor's.
        assert!((done_ms(sweep, 10 * MS, 2.0, p) - 123.3).abs() < 1e-9);
        // Later than 1.5 periods after the sweep ended is held at the top.
        assert_eq!(done_ms(sweep, 300 * MS, 1.0, p), PAIRING_DONE_TOP * p);
        // On the zoom the same finishes are ms after the sweep ended, held
        // inside its window either side.
        assert!((zoom_ms(9_100_000, 1.0, p) - 9.1).abs() < 1e-9);
        assert!((zoom_ms(-4_600_000, 1.0, p) + 4.6).abs() < 1e-9);
        assert!((zoom_ms(10 * MS, 2.0, p) - 20.0).abs() < 1e-9);
        let (early, late) = SWEEP_END_ZOOM;
        assert_eq!(zoom_ms(150 * MS, 1.0, p), late * p);
        assert_eq!(zoom_ms(-40 * MS, 1.0, p), early * p);
    }

    #[test]
    fn every_lane_has_its_states_and_colours() {
        for path in LANE_PATHS {
            let lane = lane_style(path).unwrap_or_else(|| panic!("{path} is a lane with no style"));
            assert_eq!(
                lane.values.len(),
                lane.colors.len(),
                "{path}: a state without a colour"
            );
            let mut seen = lane.values.to_vec();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), lane.values.len(), "{path}: a state twice");
        }
        // The table is the dashboard's list of lanes, and nothing else.
        for name in LANE_NAMES {
            assert!(LANE_PATHS.contains(&lane_path(name).as_str()), "{name}");
        }
        assert_eq!(LANE_PATHS.len(), LANE_NAMES.len());
        assert!(lane_style("latency/stages/1_reduce").is_none());
        assert!(lane_style("answer/line").is_none());
        assert!(lane_style("lanes/not_a_lane").is_none());
        // Health is the dark green on every lane, and a loss is red.
        for lane in [&LANE_QUEUE, &LANE_LIDAR, &LANE_CAMERA, &LANE_PAIRING] {
            assert_eq!(lane.colors[0], C_LANE_OK);
            assert_eq!(lane.colors[lane.colors.len() - 1], C_FAIL);
        }
        // A gap in a source is grey on the lanes that can show one -- neither
        // health nor the pipeline's failure -- and the lidar's and camera's
        // lanes and the pairing lane are the ones that can.
        for (lane, word) in [
            (&LANE_LIDAR, NO_SWEEP),
            (&LANE_CAMERA, NO_FRAME),
            (&LANE_PAIRING, SOURCE_GAP),
        ] {
            let i = lane.values.iter().position(|v| *v == word).unwrap();
            assert_eq!(lane.colors[i], C_ABSENT, "{word}");
        }
        assert!(!LANE_QUEUE.values.contains(&NO_SWEEP));
        assert_eq!(
            lane_style("lanes/lidar_queue").map(|l| l.values),
            Some(LANE_LIDAR.values)
        );
        assert_eq!(
            lane_style("lanes/camera_queue").map(|l| l.values),
            Some(LANE_CAMERA.values)
        );
    }

    #[test]
    fn a_lane_s_words_are_states_it_can_show() {
        // A queue's row: dropped is red, a delivery after a gap amber, and
        // a delivery that follows its predecessor clean.
        for (outcome, gap, want) in [
            (Outcome::Delivered, None, None),
            (Outcome::Delivered, Some(1.0), None),
            (Outcome::Delivered, Some(3.0), Some(("skipping", 1))),
            (Outcome::DroppedOldest, None, Some(("dropping", 2))),
            (Outcome::DroppedNewest, None, Some(("dropping", 2))),
            (Outcome::Timeout, None, Some(("dropping", 2))),
            (Outcome::Missing, None, Some(("dropping", 2))),
        ] {
            let event = queue_event(outcome, gap);
            assert_eq!(event, want, "{outcome:?} {gap:?}");
            if let Some((word, _)) = event {
                assert!(LANE_QUEUE.values.contains(&word), "{word} has no colour");
            }
        }
    }

    #[test]
    fn the_pairing_lane_reads_the_fusion_s_own_row() {
        use pipes_core::clock::{SensorTime, Tov};
        use pipes_core::sample::StreamId;
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        // The expired set's row, exactly as `track_thread` writes it: every
        // way a set expires is `expired`, and the WARN log names which.
        let mut row = Evidence::driver_missing(
            &ctx,
            StreamId::TRACKS,
            "track",
            "track",
            7,
            Tov::Time(SensorTime(700)),
            Some(HostTime(690)),
            HostTime(700),
            "pair_late",
        );
        for reason in ["pair_late", "pair_dropped", "pair_absent"] {
            row.reason = reason;
            assert_eq!(pairing_event(&row), Some(("expired", 2)));
        }
        // The produced row: `reason` blank on a clean pair, `stale` under
        // `--pair-stale-ms`.
        row.outcome = Outcome::Delivered;
        row.reason = "";
        assert_eq!(pairing_event(&row), None);
        row.reason = "stale";
        assert_eq!(pairing_event(&row), Some(("stale", 1)));
        // A set that expired because the camera's SOURCE has no frame for its
        // sweep is the source's gap, grey, not a red expiry.
        row.outcome = Outcome::Missing;
        row.reason = "pair_absent_in_source";
        assert_eq!(pairing_event(&row), Some((SOURCE_GAP, 2)));
        for word in ["stale", "expired", SOURCE_GAP] {
            assert!(LANE_PAIRING.values.contains(&word));
        }
    }

    #[test]
    fn a_sticky_lane_logs_only_its_changes_and_holds_the_worst() {
        let mut lane = Sticky::default();
        // The first row logs the lane's state, even a clean one; the next
        // clean rows log nothing.
        assert_eq!(lane.observe(None), Some("ok"));
        assert_eq!(lane.observe(None), None);
        // A drop turns it red; the skip that follows each drop does not turn
        // it amber -- a queue dropping every other frame is one red block,
        // not an alternation that shatters the lane into "N states".
        assert_eq!(lane.observe(Some(("dropping", 2))), Some("dropping"));
        for _ in 0..20 {
            assert_eq!(lane.observe(Some(("skipping", 1))), None);
            assert_eq!(lane.observe(Some(("dropping", 2))), None);
        }
        // Nine clean rows are not enough; the tenth turns it green.
        for _ in 1..LANE_CLEAR_AFTER {
            assert_eq!(lane.observe(None), None);
        }
        assert_eq!(lane.observe(None), Some("ok"));
        // A lesser failure shows when nothing worse is showing, and a worse
        // one replaces it.
        assert_eq!(lane.observe(Some(("skipping", 1))), Some("skipping"));
        assert_eq!(lane.observe(Some(("dropping", 2))), Some("dropping"));
        // A worse state that stops recurring steps down to the lesser
        // failure still happening, then to ok once that stops too.
        for _ in 1..LANE_CLEAR_AFTER {
            assert_eq!(lane.observe(Some(("skipping", 1))), None);
        }
        assert_eq!(lane.observe(Some(("skipping", 1))), Some("skipping"));
        for _ in 1..LANE_CLEAR_AFTER {
            assert_eq!(lane.observe(None), None);
        }
        assert_eq!(lane.observe(None), Some("ok"));
        // The stale run's pairing lane: sweep 0 expires (it has no older
        // frame), every sweep after it is paired stale. It reads expired for
        // ten sweeps and stale after, not expired for the whole run.
        let mut pairing = Sticky::default();
        assert_eq!(pairing.observe(Some(("expired", 2))), Some("expired"));
        let words: Vec<Option<&str>> = (0..30)
            .map(|_| pairing.observe(Some(("stale", 1))))
            .collect();
        let changed: Vec<(usize, &str)> = words
            .iter()
            .enumerate()
            .filter_map(|(i, w)| w.map(|w| (i, w)))
            .collect();
        assert_eq!(changed, [(LANE_CLEAR_AFTER as usize - 1, "stale")]);
        // A lane whose first row is a failure logs the failure first.
        assert_eq!(Sticky::default().observe(Some(("stale", 1))), Some("stale"));
    }

    #[test]
    fn a_lane_steps_down_after_a_second_of_its_own_rows() {
        // One edge feeds each lane but the fusion's, which two feed: ten rows
        // are a second on the others and half of one there.
        for camera_edge in [Some("cam0->camdet"), Some("cam0->proc")] {
            for name in LANE_NAMES {
                let want = if name == "fusion_queue" { 20 } else { 10 };
                assert_eq!(lane_clear_after(name, camera_edge), want, "{name}");
            }
        }
        // And a lane given twenty holds its failure through nineteen quiet
        // rows, stepping down on the twentieth.
        let mut lane = Sticky::new(20);
        assert_eq!(lane.observe(Some(("dropping", 2))), Some("dropping"));
        for _ in 1..20 {
            assert_eq!(lane.observe(Some(("skipping", 1))), None);
        }
        assert_eq!(lane.observe(None), Some("skipping"));
    }

    #[test]
    fn the_byte_table_says_what_each_step_does_and_where_its_result_lives() {
        let link = |i: usize, carried, built| ByteLink {
            name: CHAIN_BYTES[i].name,
            step: CHAIN_BYTES[i].step,
            carried: Some(carried),
            built: Some(built),
            read: Some(0),
            rows: 154,
            copied: 0,
        };
        let links = [
            link(0, 1_958_000, 1_964_652),
            link(1, 518_000, 519_840),
            link(2, 7_980, 11_334),
            link(3, 19_522, 27_194),
            link(4, 12_916, 16_602),
        ];
        let t = byte_table(&links, 154);
        let lines: Vec<&str> = t.lines().collect();
        assert_eq!(
            lines[0],
            "| link | what the step does | carried | change | built | read |"
        );
        assert_eq!(
            lines[2],
            "| 1 sweep | lidar driver: file into Arrow, the one copy | **1.96 MB** |  | 1.96 MB | 0 B |"
        );
        assert_eq!(
            lines[3],
            "| 2 voxels | reduce: points into 20 cm cubes | **518 kB** | ÷ 3.8 | 520 kB | 0 B |"
        );
        assert_eq!(
            lines[4],
            "| 3 dets | detect: cubes into boxes, ground removed | **8 kB** | ÷ 64.9 | 11 kB | 0 B |"
        );
        // A link that grows says so, as a product.
        assert_eq!(
            lines[5],
            "| 4 tracks | track: boxes into tracks, camera fused | **20 kB** | × 2.4 | 27 kB | 0 B |"
        );
        assert_eq!(
            lines[6],
            "| 5 answer | state: one 68-byte record per track | **13 kB** | ÷ 1.5 | 17 kB | 0 B |"
        );
        // What built and read mean: where each result lives.
        assert!(t.contains("**built** is what the step allocated"), "{t}");
        assert!(
            t.contains("**read** is what the next stage allocated"),
            "{t}"
        );
        assert!(
            t.ends_with(
                "Every link was read in place, the same `storage_id` at both ends of every row: \
                 the lidar driver's, from the file into Arrow, is the chain's one copy. \
                 Medians of 154 sweeps."
            ),
            "{t}"
        );
        // A link that allocated to read, or read a copy, is not in place, and
        // is named.
        let mut copied = links;
        copied[1].read = Some(518_000);
        copied[3].copied = 2;
        let t = byte_table(&copied, 154);
        assert!(!t.contains("Every link was read in place"), "{t}");
        assert!(t.contains("Not read in place: 2 voxels, 4 tracks."), "{t}");
        assert!(
            t.contains("| 2 voxels | reduce: points into 20 cm cubes | **518 kB** | ÷ 3.8 | 520 kB | 518 kB |"),
            "{t}"
        );
        // A run whose chain lost sweeps says what each median is over,
        // rather than calling them all medians of the driver's sweeps.
        let mut evicted = links;
        for l in &mut evicted[2..] {
            l.rows = 67;
        }
        evicted[4].rows = 63;
        let t = byte_table(&evicted, 154);
        assert!(
            t.ends_with(
                "Medians of each link's own rows: 154, 154, 67, 67, 63 for links 1 to 5, of the driver's 154 sweeps."
            ),
            "{t}"
        );
        assert!(!t.contains("Medians of 154 sweeps"), "{t}");
    }

    #[test]
    fn every_table_edge_is_one_the_tables_read() {
        // The rows the recorder keeps are exactly the ones the two tables
        // read: every link's edge and builder, and every stage's own row and
        // what it hands on, for either camera stage -- and nothing else.
        let mut read: Vec<&str> = CHAIN_BYTES
            .iter()
            .flat_map(|l| [l.edge, l.built_by])
            .chain(
                [Some("cam0->camdet"), Some("cam0->proc")]
                    .into_iter()
                    .flat_map(|c| stage_rows(c).map(|s| [s.timed_on, s.hands_on]))
                    .flatten(),
            )
            .collect();
        read.sort_unstable();
        read.dedup();
        let mut kept = TABLE_EDGES.to_vec();
        kept.sort_unstable();
        assert_eq!(read, kept);
        // The camera stage is second, and is the one that feeds the answer.
        assert_eq!(stage_rows(Some("cam0->camdet"))[1].stage, "camdet");
        assert_eq!(stage_rows(Some("cam0->proc"))[1].stage, "proc");
        for c in ["cam0->camdet", "cam0->proc"] {
            assert_eq!(stage_rows(Some(c))[1].timed_on, c);
        }
    }

    #[test]
    fn the_stage_table_says_what_each_stage_does_and_how_long_it_took() {
        let rows = stage_rows(Some("cam0->camdet"));
        let line = |i: usize, time_ns, carried| StageLine {
            row: &rows[i],
            time_ns,
            carried,
        };
        let lines = [
            line(0, Some(3_120_000), Some(1_397_406)),
            line(1, Some(37_590_000), Some(486)),
            line(3, Some(4_390_000), Some(517_940)),
            line(5, Some(120_000), None),
            line(6, Some(34_000), Some(12_973)),
        ];
        let t = stage_table(&lines, &["A note.".to_string(), "Another.".to_string()]);
        let got: Vec<&str> = t.lines().collect();
        assert_eq!(got[0], "| stage | what it does | time | hands on |");
        assert_eq!(
            got[2],
            "| cam0 | decodes each PNG into an Arrow buffer | 3.1 ms | 1.40 MB image |"
        );
        assert_eq!(
            got[3],
            "| camdet | finds objects with YOLOX-Nano, on one thread | 38 ms | 486 B of boxes |"
        );
        assert_eq!(
            got[4],
            "| reduce | averages the points into 20 cm cubes | 4.4 ms | 518 kB of cubes |"
        );
        // A stage with no output row yet says so rather than inventing one.
        assert_eq!(
            got[5],
            "| track | gives boxes ids, fuses them with the camera | 0.12 ms | - |"
        );
        assert_eq!(
            got[6],
            "| state | writes the answer, flags the nearest in path | 0.034 ms | 13 kB answer |"
        );
        // Each note its own paragraph, after the table.
        assert!(t.ends_with("\nA note.\n\nAnother.\n"), "{t:?}");
    }

    #[test]
    fn times_read_the_way_the_table_prints_them() {
        assert_eq!(human_ms(137_406_000), "137 ms");
        assert_eq!(human_ms(37_590_000), "38 ms");
        assert_eq!(human_ms(4_390_000), "4.4 ms");
        assert_eq!(human_ms(120_000), "0.12 ms");
        assert_eq!(human_ms(34_000), "0.034 ms");
        assert_eq!(human_bytes(341), "341 B");
        assert_eq!(human_bytes(12_916), "13 kB");
        assert_eq!(human_bytes(1_974_352), "1.97 MB");
        // The median of an even count is the upper one; none of nothing.
        assert_eq!(median(&[4, 1, 3, 2]), Some(3));
        assert_eq!(median(&[5]), Some(5));
        assert_eq!(median::<u64>(&[]), None);
    }

    #[test]
    fn seq_gap_skips_the_first_delivery_then_counts_frames_never_seen() {
        let mut g = SeqGap::default();
        // No predecessor: nothing to measure against.
        assert_eq!(g.observe("cam0->proc", 0), None);
        // Consecutive deliveries: the consumer kept up.
        assert_eq!(g.observe("cam0->proc", 1), Some(1.0));
        // Frames 2 and 3 were dropped ahead of the consumer.
        assert_eq!(g.observe("cam0->proc", 4), Some(3.0));
        // Edges are tracked separately, each with its own first delivery.
        assert_eq!(g.observe("cam0->rerun", 4), None);
        assert_eq!(g.observe("cam0->rerun", 5), Some(1.0));
        assert_eq!(g.observe("cam0->proc", 5), Some(1.0));
    }

    /// A dashboard for the tests that feed it rows by hand, and the viewer
    /// queue its drawings wait on: nothing draws them, so the recording is
    /// needed only for the layout the dashboard sends first.
    fn dashboard(bounds: Bounds) -> (Dashboard, Arc<ViewerQueue>) {
        let (rec, _storage) = rerun::RecordingStreamBuilder::new("pipes-test")
            .memory()
            .unwrap();
        let viewer = ViewerQueue::new(1 << 12);
        let d = Dashboard::new(&rec, viewer.canvas("rec"), bounds, Mode::File).unwrap();
        (d, viewer)
    }

    /// A delivered row on `edge` for frame `seq`, due at `due_ms` and done at
    /// `end_ms` on the host clock.
    fn timed(edge: &'static str, seq: u64, due_ms: i64, end_ms: i64) -> Evidence {
        use pipes_core::clock::{SensorTime, Tov};
        use pipes_core::sample::StreamId;
        const MS: i64 = 1_000_000;
        let mut e = Evidence::driver_missing(
            &RowCtx {
                run_id: "t".to_string(),
                t0_host: HostTime(0),
                epoch: 0,
            },
            StreamId::LIDAR,
            edge,
            "stage",
            seq,
            Tov::Time(SensorTime(due_ms * MS)),
            Some(HostTime(due_ms * MS)),
            HostTime(end_ms * MS),
            "",
        );
        e.outcome = Outcome::Delivered;
        e.proc_start_ns = Some((end_ms - 1) * MS);
        e.proc_end_ns = Some(end_ms * MS);
        e
    }

    #[test]
    fn the_camera_dot_waits_for_its_sweep_and_counts_a_late_camera() {
        let (mut d, _viewer) = dashboard(Bounds {
            camera_edge: Some("cam0->camdet"),
            period_ms: Some(100.0),
            rate: Some(1.0),
            ..Bounds::default()
        });
        // Frame 1, healthy: the camera done at 97 ms, 3 ms before its sweep
        // is due at 100, and before `detect` is done at 109. The camera's
        // row comes first; its dot waits for the sweep's.
        d.log_row(&timed("cam0->camdet", 1, 59, 97)).unwrap();
        assert_eq!(d.halves.get(&1).map(|h| h.drawn), Some(false));
        d.log_row(&timed("velo", 1, 100, 100)).unwrap();
        assert_eq!(d.halves.get(&1).map(|h| h.drawn), Some(true));
        assert_eq!(d.camera_later, (0, 0));
        d.log_row(&timed("det->detect", 1, 100, 109)).unwrap();
        assert_eq!(d.camera_later, (1, 0));
        assert!(!d.halves.contains_key(&1), "a joined sweep is kept");
        // Frame 2, a slow camera: done at 246, after the sweep (200) and
        // after `detect` (209). Counted late.
        d.log_row(&timed("velo", 2, 200, 200)).unwrap();
        d.log_row(&timed("det->detect", 2, 200, 209)).unwrap();
        d.log_row(&timed("cam0->camdet", 2, 159, 246)).unwrap();
        assert_eq!(d.camera_later, (2, 1));
        assert!(d.halves.is_empty(), "{:?}", d.halves);
        // Frame 3's camera frame was dropped: its half waits, and is let go
        // once the run is `HALVES_KEPT` frames past it.
        d.log_row(&timed("velo", 3, 300, 300)).unwrap();
        d.log_row(&timed("det->detect", 3, 300, 309)).unwrap();
        assert!(d.halves.contains_key(&3));
        for seq in 4..=(3 + HALVES_KEPT + 1) {
            let due = seq as i64 * 100;
            d.log_row(&timed("velo", seq, due, due)).unwrap();
        }
        assert!(!d.halves.contains_key(&3), "a sweep with no camera is kept");
        assert!(d.halves.len() <= HALVES_KEPT as usize + 1);
        assert_eq!(d.camera_later, (2, 1));
        // A row no half is timed on leaves no entry behind.
        let before = d.halves.len();
        d.log_row(&timed("obj->track", 900, 90_000, 90_010))
            .unwrap();
        assert_eq!(d.halves.len(), before);
    }

    #[test]
    fn the_stage_notes_say_what_the_camera_s_time_holds() {
        // The slow camera: 100 ms of sleep inside camdet's window, 80 MB of
        // working memory, 116 of 154 frames, and late on every sweep.
        let (mut d, _viewer) = dashboard(Bounds {
            camera_edge: Some("cam0->camdet"),
            camera_delay_ms: 100,
            ..Bounds::default()
        });
        let rows = |n: usize, carried: u64, alloc: u64| EdgeRows {
            carried: vec![carried; n],
            alloc: vec![alloc; n],
            time_ns: vec![1; n],
            copied: 0,
        };
        d.table.insert("cam0", rows(154, 1_397_406, 1_690_042));
        d.table
            .insert("cam0->camdet", rows(116, 1_397_406, 80_243_692));
        d.table.insert("cam_det", rows(116, 486, 6_900));
        d.camera_later = (116, 116);
        let notes = d.stage_notes(&stage_rows(Some("cam0->camdet"))[1]);
        assert_eq!(
            notes[1..],
            [
                "camdet's includes the 100 ms sleep of `--consumer-delay-ms`.",
                "camdet allocates 80.24 MB while it works on each frame, the network's working memory, and hands on 486 B.",
                "camdet ran on 116 of the 154 frames the camera delivered.",
                "The camera's result was ready after the lidar's boxes on 116 of 116 sweeps: on the Demo page, the blue diamond above `detect`'s teal one.",
            ]
        );
        assert!(notes[0].contains("track's includes any wait for the camera"));
        // The healthy run says the camera was always first, and nothing of
        // a sleep or of frames it missed.
        d.bounds.camera_delay_ms = 0;
        d.table
            .insert("cam0->camdet", rows(154, 1_397_406, 80_243_692));
        d.camera_later = (154, 0);
        let notes = d.stage_notes(&stage_rows(Some("cam0->camdet"))[1]);
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert_eq!(
            notes[2],
            "The camera's result was ready before the lidar's boxes on all 154 sweeps."
        );
    }

    /// Every shape of row the recorder mirrors, three sweeps' worth: each
    /// sensor driver's admitted frame and a missing one; each derived
    /// stream's producer row, the fusion's clean, stale and expired; and on
    /// every queue this binary opens a delivered row, a dropped one and a
    /// delivered one after the gap. Each row carries every column the
    /// dashboard reads.
    fn every_kind_of_row() -> Vec<Evidence> {
        use crate::dashboard::{ADMISSION_EDGES, EDGES};
        use pipes_core::clock::{SensorTime, Tov};
        use pipes_core::sample::StreamId;
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        const MS: i64 = 1_000_000;
        let stream = |edge: &str| -> u8 {
            let producer = edge.split("->").next().unwrap_or(edge);
            match producer {
                "cam0" => 0,
                "velo" => 1,
                "det" => 3,
                "cam_det" => 4,
                "state" => 5,
                "track" => 6,
                _ => 7,
            }
        };
        let mut rows = Vec::new();
        let mut arrival = 0i64;
        let mut row = |edge: &'static str, seq: u64, outcome: Outcome, reason: &'static str| {
            arrival += 10 * MS;
            let start = seq as i64 * 100 * MS;
            let end = if edge.starts_with("cam0") {
                start
            } else {
                start + 103 * MS
            };
            let mut e = Evidence::driver_missing(
                &ctx,
                StreamId::LIDAR,
                edge,
                "stage",
                seq,
                Tov::Range {
                    start: SensorTime(start),
                    end: SensorTime(end.max(start + 1)),
                },
                Some(HostTime(start)),
                HostTime(arrival),
                reason,
            );
            e.tov_end_ns = end;
            e.stream = stream(edge);
            e.outcome = outcome;
            if ADMISSION_EDGES.contains(&edge) && outcome == Outcome::Delivered {
                e.arrival_seq = Some(arrival as u64);
            }
            if edge.contains("->") {
                e.depth_at_push = Some(if outcome == Outcome::Delivered { 1 } else { 4 });
            }
            if outcome == Outcome::Delivered {
                e.depth_after_pop = edge.contains("->").then_some(0);
                e.proc_start_ns = Some(arrival);
                e.proc_end_ns = Some(arrival + 5 * MS);
                e.queue_wait_ns = Some(2 * MS);
                e.measurement_age_ns = Some(40 * MS);
                e.storage_id = 1000 + seq as usize;
                e.payload_bytes = 1000;
            }
            rows.push(e);
        };
        for seq in 0..3u64 {
            for edge in ADMISSION_EDGES {
                row(edge, seq, Outcome::Delivered, "");
            }
            for edge in ["det", "obj", "cam_det", "state"] {
                row(edge, seq, Outcome::Delivered, "");
            }
            match seq {
                0 => row("track", seq, Outcome::Delivered, ""),
                1 => row("track", seq, Outcome::Delivered, "stale"),
                _ => row("track", seq, Outcome::Missing, "pair_late"),
            }
            for edge in EDGES {
                if seq == 1 {
                    row(edge, seq, Outcome::DroppedOldest, "evicted");
                } else {
                    row(edge, seq, Outcome::Delivered, "");
                }
            }
        }
        for edge in ADMISSION_EDGES {
            row(edge, 3, Outcome::Missing, "absent");
        }
        // A frame each sensor's source never had: no instant, and the
        // fusion's expiry for the camera's.
        for edge in ADMISSION_EDGES {
            row(edge, 4, Outcome::Missing, "absent_in_source");
        }
        row("track", 4, Outcome::Missing, "pair_absent_in_source");
        for e in rows.iter_mut().filter(|e| e.reason == "absent_in_source") {
            e.tov_start_ns = 0;
            e.tov_end_ns = 0;
        }
        rows
    }

    #[test]
    fn every_series_the_dashboard_logs_is_styled_and_in_a_view() {
        // Read back from what the dashboard LOGGED, not from a list kept by
        // hand beside it: a leaf added to `log_row` without a style, or
        // without a view to show it, fails here the first time it is logged.
        use crate::dashboard::{logged, send_blueprint};
        use rerun::{RecordingStreamBuilder, StoreKind};
        for camera_edge in ["cam0->camdet", "cam0->proc"] {
            let bounds = Bounds {
                image_wh: Some((1242.0, 375.0)),
                period_ms: Some(103.3),
                camera_edge: Some(camera_edge),
                camera_delay_ms: 50,
                rate: Some(1.0),
                lidar_lane: true,
                answer: true,
            };
            let (rec, storage) = RecordingStreamBuilder::new("pipes-test").memory().unwrap();
            let viewer = ViewerQueue::new(1 << 16);
            let mut d =
                Dashboard::new(&rec, viewer.canvas("rec"), bounds.clone(), Mode::File).unwrap();
            for e in every_kind_of_row() {
                d.log_row(&e).unwrap();
            }
            let ctx = RowCtx {
                run_id: "t".to_string(),
                t0_host: HostTime(0),
                epoch: 0,
            };
            d.log_event(&Event::recorder_degraded(&ctx, HostTime(1), "proc", 2))
                .unwrap();
            d.log_event(&Event::shutdown(&ctx, HostTime(2), 3, "done"))
                .unwrap();
            d.finish().unwrap();
            // What the viewer thread would do with every drawing queued.
            assert_eq!(viewer.dropped(), 0, "the test's queue overflowed");
            viewer.draw_queued(&rec);
            rec.flush_blocking().unwrap();
            let logged = logged(&storage, StoreKind::Recording);
            let (bp, _bp_storage) = RecordingStreamBuilder::new("pipes-test").memory().unwrap();
            let layout = send_blueprint(&bp, Mode::File, &bounds).unwrap();

            // The run under test drew every family the dashboard has.
            for want in [
                "lanes/pairing",
                "pairing/1_camera_done",
                "pairing/3_detect_done",
                "pairing/4_answer_done",
                "after_sweep/sweep_end",
                "after_sweep/1_camera_done",
                "after_sweep/2_reduce_done",
                "after_sweep/3_detect_done",
                "after_sweep/4_answer_done",
                "latency/table",
                "bytes/table",
                "graph/pipeline",
                "log/drops",
                "log/events",
                // The gaps: each sensor's lane, and what was cleared or
                // stamped for the frame that is not there.
                "lanes/lidar_queue",
                "lanes/camera_queue",
                "camera/status",
                "answer/headline",
                "lidar/sweep",
            ] {
                assert!(
                    logged.contains_key(want),
                    "{camera_edge}: nothing at {want}"
                );
            }
            // And nothing of the pages that were cut.
            for path in logged.keys() {
                for gone in [
                    "queues/",
                    "latency/headroom",
                    "latency/stages",
                    "bytes/chain",
                    "bytes/alloc",
                ] {
                    assert!(!path.starts_with(gone), "{camera_edge}: {path} is logged");
                }
            }
            let has = |comps: &BTreeSet<String>, archetype: &str| {
                comps
                    .iter()
                    .any(|c| c.starts_with(&format!("{archetype}:")))
            };
            for (path, comps) in &logged {
                if path.starts_with("__") {
                    continue; // the SDK's own recording properties
                }
                assert!(
                    layout.shows(path),
                    "{camera_edge}: {path} is logged but no view shows it"
                );
                if has(comps, "Scalars") {
                    assert!(
                        has(comps, "SeriesLines") || has(comps, "SeriesPoints"),
                        "{camera_edge}: {path} is a series with no name or colour: {comps:?}"
                    );
                    // And its leaf is in the table the legend tests walk.
                    let leaf = path.rsplit('/').next().unwrap_or(path);
                    assert!(
                        SERIES_LEAVES.contains(&leaf),
                        "{camera_edge}: {leaf} ({path}) is not in SERIES_LEAVES"
                    );
                }
                if has(comps, "Measurements") {
                    assert!(
                        comps.iter().any(|c| c.starts_with("Measurements:names")),
                        "{camera_edge}: {path} is a band with no name: {comps:?}"
                    );
                }
                if has(comps, "StateChange") {
                    assert!(
                        has(comps, "StateConfiguration"),
                        "{camera_edge}: {path} is a lane with no states' colours: {comps:?}"
                    );
                }
            }
        }
    }

    /// The seqs a stream carries are the frames of the sensors it counts,
    /// so a frame absent in a sensor's source is absent on every edge of
    /// every such stream -- the camera's on `cam_det->track` as on
    /// `cam0->camdet` -- and never on a stream of the other sensor.
    #[test]
    fn a_source_gap_is_stepped_over_on_every_stream_that_counts_its_frames() {
        use pipes_core::sample::StreamId;
        let (mut d, _viewer) = dashboard(Bounds::default());
        d.absent.insert((Sensor::Camera, 8));
        d.absent.insert((Sensor::Lidar, 3));
        d.absent.insert((Sensor::Lidar, 4));
        let row = |stream: StreamId, seq: u64| {
            let mut e = Evidence::driver_missing(
                &RowCtx {
                    run_id: "t".to_string(),
                    t0_host: HostTime(0),
                    epoch: 0,
                },
                stream,
                "e",
                "s",
                seq,
                pipes_core::clock::Tov::None,
                None,
                HostTime(0),
                "",
            );
            e.stream = stream.0;
            e
        };
        // 7 -> 9 over the camera's frame 8, on the camera and on cam_det.
        assert_eq!(d.absent_between(&row(StreamId::CAM0, 9), 2.0), 1.0);
        assert_eq!(d.absent_between(&row(StreamId::CAM_DET, 9), 2.0), 1.0);
        // The same step on the lidar is the lidar's own: nothing to excuse.
        assert_eq!(d.absent_between(&row(StreamId::LIDAR, 9), 2.0), 0.0);
        // 2 -> 5 over the lidar's frames 3-4, on the lidar and every stream
        // derived from its sweeps -- and on nothing of the camera's.
        for s in [StreamId::LIDAR, StreamId::LIDAR_DET, StreamId::LIDAR_OBJ] {
            assert_eq!(d.absent_between(&row(s, 5), 3.0), 2.0, "{s:?}");
        }
        assert_eq!(d.absent_between(&row(StreamId::CAM_DET, 5), 3.0), 0.0);
        // The fusion's output lacks an instant either source lacks.
        assert_eq!(d.absent_between(&row(StreamId::TRACKS, 9), 7.0), 3.0);
        assert_eq!(d.absent_between(&row(StreamId::EGO, 9), 7.0), 3.0);
        // A consecutive delivery steps over nothing.
        assert_eq!(d.absent_between(&row(StreamId::CAM0, 8), 1.0), 0.0);
    }
}
