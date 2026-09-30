//! The dashboard's entity tree and the layout that shows it, in one table.
//!
//! Everything a run draws onto its Rerun recording hangs off the paths named
//! here, and the layout is a set of path prefixes and nothing else -- a view's
//! contents are `+ /camera/image`-style expressions -- so the tree IS the
//! layout. A stage that logs a picture takes its path from this module rather
//! than spelling one, and a path spelled nowhere else cannot drift from the
//! view that shows it; a test walks every view's contents against the entity
//! table.
//!
//! Naming: `camera/` and `lidar/` are the two sensors' pictures (the frame,
//! and what the chain drew on it or found in it); `answer/` is the end of the
//! chain, as a headline, a time to collision and a sentence; `pairing/` is the
//! sweep's window, the camera frame paired inside it, and when the camera's
//! half, the lidar's half and the answer were ready; `after_sweep/` the same
//! finishes and `reduce`'s, zoomed onto the sweep's end; `tracks/` the
//! fusion's populations in the frame and the share of each sensor's it fused;
//! `lanes/` the five sticky health lanes, the pairing's and the queues';
//! `latency/` each stage's time as a table; `bytes/` the byte chain as a
//! table; `graph/` the pipeline; `log/` the WARN rows and the run's events. Every entity is shown by some view and every view shows an entity,
//! and a test holds both. Nothing hangs under `pipes/` any more: that prefix
//! grouped 147 entities by which thread wrote them, which is not what a
//! reader asks.
//!
//! The layout ([`send_blueprint`]) is built by hand as a blueprint store and
//! sent in-band on the run's own stream before the first data row, so a
//! viewer the run spawns opens straight into it and an `.rrd` file carries it
//! at its head. The typed `rerun::blueprint` API cannot attach the properties
//! that carry the design -- the locked y-axes, the hidden legends, the visible
//! time window, the 3D eye -- and an `.rbl` file passed on the command line
//! does not apply. So each view is logged as the SDK's own `Blueprint::send`
//! logs it, plus one archetype per property at `view/<id>/<Archetype>`.

use rerun::blueprint::components::{
    ActiveTab, AutoLayout, AutoViews, BackgroundKind, ContainerKind, Enabled, Eye3DKind,
    IncludedContent, LinkAxis, LoopMode, PanelState, PlayState, QueryExpression, RootContainer,
    TextLogColumn, TimelineColumn, ViewClass, VisibleTimeRange,
};
use rerun::blueprint::encodings::{
    TextLogColumn as TextLogColumnSpec, TextLogColumnKind, TimelineColumn as TimelineColumnSpec,
};
use rerun::components::{Color, Plane3D, Position3D, TextLogLevel};
use rerun::external::re_log_types::{BlueprintActivationCommand, RecordingId};
use rerun::external::re_sdk_types::blueprint::archetypes::{
    Background, ContainerBlueprint, EntityBehavior, EyeControls3D, ForceCenter,
    ForceCollisionRadius, ForceLink, ForceManyBody, ForcePosition, LineGrid3D, PanelBlueprint,
    PlotLegend, ScalarAxis, TextLogColumns, TextLogRows, TimeAxis, TimePanelBlueprint,
    ViewBlueprint, ViewContents, ViewportBlueprint, VisibleTimeRanges, VisualBounds2D,
};
use rerun::external::re_sdk_types::encodings::{
    Bool, Range1D, Range2D, TimeInt, TimeRange, TimeRangeBoundary, Uuid,
    VisibleTimeRange as VisibleTimeRangeSpec,
};
use rerun::{AsComponents, RecordingStream, RecordingStreamBuilder, RecordingStreamError};

use crate::record::Bounds;

/// Entity paths, without a leading slash (the SDK's form).
pub mod entity {
    /// The camera's root: the annotation context that names the fusion's
    /// populations by class id is logged here, for everything under it.
    pub const CAMERA_ROOT: &str = "camera";
    /// The lidar's root, for the same annotation context.
    pub const LIDAR_ROOT: &str = "lidar";
    /// The camera frame the `rerun` consumer received, as it was given.
    pub const CAMERA_IMAGE: &str = "camera/image";
    /// Every record of the answer that is in the paired camera frame, and
    /// every camera-only detection it carries, coloured and class-id'd by
    /// POPULATION (the annotation context at `camera/`: 1 fused, 2
    /// lidar-only, 3 camera-only, 9 answer): a box each, and, on the fused
    /// and camera-only ones, a point on the box's bottom (fused) or top
    /// (camera-only) edge carrying the label (`car 0.91 · 11.4 m · 3.7 m/s ·
    /// 1.8 s`, `car 0.62 · no range`) -- a box's own label wraps at the box's
    /// width in the viewer; a point's does not. Drawn by `state` from the
    /// answer's own records.
    pub const CAMERA_TRACKS: &str = "camera/tracks";
    /// One line in the image's top-left corner saying which frame the boxes
    /// were fused with and how old it was: `cam 76 · paired +11 ms`, or, in
    /// amber, `STALE · cam 75 · -93 ms`. Drawn by `state` on every paired
    /// frame -- and by `track`, in red, `EXPIRED · sweep 57 · pair_dropped`,
    /// on a sweep whose set expired and so has no answer (`NO FRAME · sweep
    /// 30 · pair_absent_in_source` when the camera's source has no frame of
    /// that number); by the recorder, `NO FRAME` or `NO SWEEP · frame N ·
    /// absent_in_source`, on the frame a sensor's source never had.
    pub const CAMERA_STATUS: &str = "camera/status";
    /// The one record the answer flags nearest in the path, on the same
    /// picture: a thick box, class 9, with no label of its own.
    pub const CAMERA_ANSWER: &str = "camera/answer";
    /// The answer's label on the picture, on its own entity: a point at the
    /// middle of the answer box's bottom edge, kept on the image (x at least
    /// 190 px from either side, y inside it) so a box at the frame's edge
    /// does not push its words off the picture. `car 0.91 · 11.4 m · TTC 3.1
    /// s`, or `11.4 m · closing 3.7 m/s · TTC 3.1 s` without a class; no id.
    pub const CAMERA_ANSWER_LABEL: &str = "camera/answer_label";
    /// What the frozen camera detector found in the frame, drawn by `camdet`
    /// on the frame it ran on: a purple box each, labelled `class conf` on a
    /// point on its top edge. In the camera view but hidden by default (the
    /// view's eye toggle shows it): the fused and camera-only boxes already
    /// draw every one of them.
    pub const CAMERA_DET: &str = "camera/cam_det";
    /// The raw sweep the `cloud` stage read.
    pub const LIDAR_SWEEP: &str = "lidar/sweep";
    /// The voxel cloud `reduce` built.
    pub const LIDAR_VOXELS: &str = "lidar/voxels";
    /// A box round every detection: in the raw-sweep view, hidden by
    /// default, because at a sweep's density its boxes and the tracks'
    /// coincide.
    pub const LIDAR_DETECTIONS: &str = "lidar/detections";
    /// The same boxes one link later, each with an identity.
    pub const LIDAR_TRACKS: &str = "lidar/tracks";
    /// The one object the answer flags, as a gold wireframe among the tracks.
    pub const LIDAR_ANSWER: &str = "lidar/answer";
    /// Root of the answer's series (`answer/ttc_s`, `answer/ttc_warn`) and
    /// its sentence (`answer/line`, one per sweep: the answer's, or why the
    /// sweep had none).
    pub const ANSWER: &str = "answer";
    /// The answer as the demo screen's headline: one markdown heading per
    /// sweep, `## 11.4 m ahead · closing 3.7 m/s · TTC 3.1 s · answer 26 ms
    /// old`, led by the camera's class when it has one and ending in `STALE
    /// camera 93 ms` on a stale pair. Drawn by `state-sink`, and by `track`
    /// on a sweep whose set expired: `## no answer for sweep 57 · camera
    /// frame dropped (pair_dropped)`.
    pub const ANSWER_HEADLINE: &str = "answer/headline";
    /// Root of the pairing's own numbers ([`super::PAIRING_LEAVES`] and
    /// [`super::PAIRING_BAND`]): the sweep's range as a band with its two
    /// ends, its trigger, and the instant of the camera frame paired with it,
    /// in ms from the sweep's start -- and, on the same axis, when the
    /// pairing's two halves and its answer were ready
    /// ([`super::READY_DOTS`]).
    pub const PAIRING: &str = "pairing";
    /// The zoom on the sweep's end, in ms after the sweep ended: the end
    /// itself as a line at 0, and every ready dot, `reduce`'s among them
    /// ([`super::READY_DOTS`]).
    pub const AFTER_SWEEP: &str = "after_sweep";
    /// Root of the five sticky health lanes, `lanes/<name>`
    /// ([`super::LANE_NAMES`]): one state each, logged only when it changes.
    pub const LANES: &str = "lanes";
    /// Every stage as a table -- what it does, its median time per sample and
    /// what it hands on -- from the medians of the run so far: written once
    /// after the first ten answers and again at the end.
    pub const STAGE_TABLE: &str = "latency/table";
    /// The byte chain as a table -- each link, what the step that made it
    /// does, its carried bytes, the change from the link before, what the
    /// step allocated to build it and what the next stage allocated to read
    /// it -- from the medians of the run so far, written when the stage
    /// table is.
    pub const BYTES_TABLE: &str = "bytes/table";
    /// The pipeline as a graph, each node labelled with the bytes its stage
    /// hands on for the latest sweep: the edges once, statically, and the
    /// nodes per answer ([`super::GRAPH_NODES`]).
    pub const GRAPH_PIPELINE: &str = "graph/pipeline";
    /// One WARN row per dropped or missing sample, for every edge: the only
    /// place a drop's REASON appears.
    pub const LOG_DROPS: &str = "log/drops";
    /// Run-level events (recorder degraded, shutdown).
    pub const LOG_EVENTS: &str = "log/events";
    /// Root of the fusion's population counts in the camera frame, one leaf
    /// each ([`super::TRACK_COUNT_LEAVES`]).
    pub const TRACKS_COUNT: &str = "tracks/count";
    /// The share of the in-frame lidar tracks the camera confirmed, 0 to 1,
    /// logged by `state-sink` on every answer with a lidar track in frame.
    pub const TRACKS_FUSED_FRACTION: &str = "tracks/fused_fraction";
    /// The same from the camera's side: the share of the camera's detections
    /// in the frame the lidar confirmed, logged on every answer with a
    /// detection in it.
    pub const TRACKS_CAMERA_FUSED_FRACTION: &str = "tracks/camera_fused_fraction";
}

/// Every real queue this binary can open, in chain order, as the edge name
/// the evidence rows carry: what the health lanes are fed from
/// ([`queue_lane`]), and the table the chart tests check their edges against.
pub const EDGES: [&str; 12] = [
    "cam0->proc",
    "cam0->rerun",
    "cam0->camdet",
    "velo->cloud",
    "velo->reduce",
    "det->cloud",
    "det->detect",
    "obj->sink",
    "obj->track",
    "cam_det->track",
    "track->state",
    "state->sink",
];

/// The two sensor drivers' pseudo-edges: rows for frames and sweeps that were
/// admitted, or never produced at all. The only producer rows that can be
/// `Missing` -- a derived stream's producer row (`det`, `obj`, `track`,
/// `state`, `cam_det`) is structurally always Delivered -- and so the only
/// rows that can say a frame is absent in the source.
pub const ADMISSION_EDGES: [&str; 2] = ["cam0", "velo"];

/// One link of the byte chain, a row of [`entity::BYTES_TABLE`].
#[derive(Clone, Copy, Debug)]
pub struct ChainLink {
    /// What the link carries: `sweep`, `voxels`, ...
    pub name: &'static str,
    /// The edge into the next stage of the chain: its delivered
    /// `payload_bytes` are what the link carried, and its `bytes_alloc`
    /// what that stage allocated to read it.
    pub edge: &'static str,
    /// The producer row of the step that made it: its `bytes_alloc` is what
    /// the step allocated to build it.
    pub built_by: &'static str,
    /// What that step does, for a reader who has not met the pipeline.
    pub step: &'static str,
}

/// The byte chain, in chain order: the sweep, the voxels, the detections,
/// the tracks, the answer, each as the next stage of the chain read it.
pub const CHAIN_BYTES: [ChainLink; 5] = [
    ChainLink {
        name: "sweep",
        edge: "velo->reduce",
        built_by: "velo",
        step: "lidar driver: file into Arrow, the one copy",
    },
    ChainLink {
        name: "voxels",
        edge: "det->detect",
        built_by: "det",
        step: "reduce: points into 20 cm cubes",
    },
    ChainLink {
        name: "dets",
        edge: "obj->track",
        built_by: "obj",
        step: "detect: cubes into boxes, ground removed",
    },
    ChainLink {
        name: "tracks",
        edge: "track->state",
        built_by: "track",
        step: "track: boxes into tracks, camera fused",
    },
    ChainLink {
        name: "answer",
        edge: "state->sink",
        built_by: "state",
        step: "state: one 68-byte record per track",
    },
];

/// The pipeline graph's nodes, as `(id, x, y)` in the graph's own units:
/// the two sensors on the left, the admission, the camera stage above the
/// lidar's two, the fusion, the answer's state, and the answer on the right.
/// Fixed positions, every force off, so the picture is the same every sweep.
/// The rows are 70 apart, not 40: the camera, the admission and the camera
/// stage carry a second line saying what they do, and at 40 a two-line
/// label met its diagonal neighbour's.
pub const GRAPH_NODES: [(&str, f32, f32); 9] = [
    ("cam0", 0.0, -70.0),
    ("velo", 0.0, 70.0),
    ("admit", 120.0, 0.0),
    ("camera", 240.0, -70.0),
    ("reduce", 240.0, 70.0),
    ("detect", 360.0, 70.0),
    ("track", 480.0, 0.0),
    ("state", 600.0, 0.0),
    ("answer", 720.0, 0.0),
];

/// The pipeline graph's edges, directed, between [`GRAPH_NODES`] ids.
pub const GRAPH_EDGES: [(&str, &str); 9] = [
    ("cam0", "admit"),
    ("velo", "admit"),
    ("admit", "camera"),
    ("admit", "reduce"),
    ("camera", "track"),
    ("reduce", "detect"),
    ("detect", "track"),
    ("track", "state"),
    ("state", "answer"),
];

/// What times one of the pairing plot's ready dots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageClock {
    /// `proc_end - due` on this edge's delivered row: the stage that reads
    /// the edge finished with the sweep that long after the sweep was due.
    Edge(&'static str),
    /// The camera queue that feeds the answer ([`crate::record::Bounds`]'s
    /// `camera_edge`): when its stage finished with the camera frame of the
    /// sweep's number, against the SWEEP's due. The camera row's own `due`
    /// is its frame's instant, mid-sweep, so that row alone cannot say it.
    Camera,
}

/// One ready dot: when a stage finished with a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadyDot {
    /// Its leaf, under `pairing/` and under `after_sweep/`, and so its style.
    pub leaf: &'static str,
    pub clock: StageClock,
    /// Whether the pairing plot draws it, or only the zoom on the sweep's
    /// end.
    pub overview: bool,
}

/// When the stages the pairing waits on finished with a sweep: the camera's
/// half (its stage done with the frame of the sweep's number), the lidar's
/// two stages, `reduce` and `detect`, and the answer (`state-sink` done).
/// The pairing plot draws them in ms from the sweep's start, beside the
/// camera frame's instant, and leaves out `reduce`, which is inside the
/// lidar's half there; the zoom on the sweep's end draws all four in ms after
/// the sweep ended. A camera dot above `detect`'s is a camera frame that was
/// not ready when the lidar's boxes were, so the fusion waited for it or did
/// without it. `track` and `state` finish within a millisecond of the
/// answer, and are left out of both. Numbered in the order they are drawn,
/// last on top, so the answer's gold sits over `detect`'s teal when the two
/// coincide.
pub const READY_DOTS: [ReadyDot; 4] = [
    ReadyDot {
        leaf: "1_camera_done",
        clock: StageClock::Camera,
        overview: true,
    },
    ReadyDot {
        leaf: "2_reduce_done",
        clock: StageClock::Edge("velo->reduce"),
        overview: false,
    },
    ReadyDot {
        leaf: "3_detect_done",
        clock: StageClock::Edge("det->detect"),
        overview: true,
    },
    ReadyDot {
        leaf: "4_answer_done",
        clock: StageClock::Edge("state->sink"),
        overview: true,
    },
];

/// The highest a ready dot is drawn on the pairing plot, in sweep periods
/// from the sweep's start: a half or an answer later than that -- 1.5
/// periods after the sweep ended -- is drawn here, on the plot's top edge,
/// as "this late or later".
pub const PAIRING_DONE_TOP: f64 = 2.5;

/// The zoom on the sweep's end, in sweep periods after the sweep ended: the
/// edges a ready dot is held at, from 0.15 of a period before the end -- a
/// camera frame ready that early was ready in time -- to 0.25 after it,
/// past the healthy run's answer at about a tenth. A dot outside is drawn at
/// the edge, as "this early or earlier" or "this late or later"; the pairing
/// plot above says how late.
pub const SWEEP_END_ZOOM: (f64, f64) = (-0.15, 0.25);

/// The zoom's reference line, `after_sweep/sweep_end`: the sweep's end, at
/// 0. The pairing plot's own `sweep_end` leaf, and so its style.
pub const SWEEP_END_LEAF: &str = "sweep_end";

/// The answer's scalar leaves under `answer/`, as `state-sink` draws them:
/// `ttc_s`, the flagged object's time to contact, logged only while it
/// closes and drawn at 10 when it is 10 s or more, so a receding object is a
/// gap rather than a TTC of 0; and `ttc_warn`, the 3 s line it is read
/// against, on every answer. The distance and the closing speed are words in
/// the label and the headline, not lines. Named here, beside the entity
/// table, and styled through `record::series_archetype`: a leaf spelled here
/// and not there is a failing test rather than an anonymous grey line in
/// somebody's recording.
pub const ANSWER_LEAVES: [&str; 2] = ["ttc_s", "ttc_warn"];

/// The time to contact at which the TTC plot's warning line sits, seconds.
pub const TTC_WARN_S: f64 = 3.0;

/// The most the TTC plot draws, seconds: a longer time to contact is drawn
/// here, at the plot's top edge, as "this much or more".
pub const TTC_TOP_S: f64 = 10.0;

/// The fusion's populations in the camera frame, under `tracks/count/`,
/// logged on EVERY answer by `state-sink`: records the camera confirmed,
/// in-frame records only the lidar holds, and detections only the camera
/// holds. Styled through `record::series_archetype`, like every leaf.
pub const TRACK_COUNT_LEAVES: [&str; 3] = ["fused", "lidar_only", "camera_only"];

/// The pairing's scalar leaves under `pairing/`, which the `track` stage
/// logs itself for every sweep because no evidence column carries them, in
/// ms from the sweep's start on the sensor clock: the range's two ends, its
/// trigger, and the instant of the camera frame the fusion paired with it.
/// Styled through `record::series_archetype`, like every leaf.
pub const PAIRING_LEAVES: [&str; 4] = ["sweep_start", "sweep_end", "trigger", "camera"];

/// The leaf, and so the style, of [`entity::TRACKS_FUSED_FRACTION`].
pub const FUSED_FRACTION_LEAF: &str = "fused_fraction";

/// The leaf, and so the style, of [`entity::TRACKS_CAMERA_FUSED_FRACTION`].
pub const CAMERA_FUSED_FRACTION_LEAF: &str = "camera_fused_fraction";

/// The pairing's band, `pairing/window`: a grey `Measurements` of half the
/// sweep's period with that half as its standard deviation, so it spans the
/// range the camera frame should sit in.
pub const PAIRING_BAND: &str = "window";

/// The five sticky health lanes, as `lanes/<name>`, on the Fusion page: the
/// pairing, then the queues in chain order -- the camera queue that feeds
/// the answer (`cam0->camdet` with the detector, `cam0->proc` without), the
/// lidar's (`velo->reduce`), the detector's (`det->detect`) and the fusion's
/// two inputs (`cam_det->track` and `obj->track`) as one.
pub const LANE_NAMES: [&str; 5] = [
    LANE_PAIRING_NAME,
    LANE_CAMERA_NAME,
    LANE_LIDAR_NAME,
    "detect_queue",
    "fusion_queue",
];

/// The pairing's lane, fed by the fusion's own row -- and by the velodyne
/// driver's row for a frame whose sweep is absent in the source, which the
/// fusion never sees; a missing camera frame reaches it through the fusion's
/// own expired row.
pub const LANE_PAIRING_NAME: &str = "pairing";

/// The lidar's lane, fed by `velo->reduce` -- and by the velodyne driver's
/// row for a frame whose sweep is absent in the source.
pub const LANE_LIDAR_NAME: &str = "lidar_queue";

/// The camera's lane, fed by the camera queue that feeds the answer -- and by
/// the camera driver's row for a frame whose PNG is absent in the source.
pub const LANE_CAMERA_NAME: &str = "camera_queue";

/// One lane's entity path.
pub fn lane_path(name: &str) -> String {
    format!("{}/{name}", entity::LANES)
}

/// Which lane a queue's rows feed, given the camera queue that feeds the
/// answer, or `None` for an edge no lane watches.
pub fn queue_lane(edge: &str, camera_edge: Option<&str>) -> Option<&'static str> {
    if camera_edge == Some(edge) {
        return Some(LANE_CAMERA_NAME);
    }
    match edge {
        "velo->reduce" => Some(LANE_LIDAR_NAME),
        "det->detect" => Some("detect_queue"),
        "cam_det->track" | "obj->track" => Some("fusion_queue"),
        _ => None,
    }
}

/// Every entity a run can log, as the tree the layout is checked against.
/// The per-bar and per-leaf groups are generated from the lists the stages
/// log by, so a new bar or leaf is in the table the moment it is in its list.
#[cfg(test)]
pub fn entities() -> Vec<String> {
    let mut all: Vec<String> = [
        entity::CAMERA_ROOT,
        entity::LIDAR_ROOT,
        entity::CAMERA_IMAGE,
        entity::CAMERA_TRACKS,
        entity::CAMERA_STATUS,
        entity::CAMERA_ANSWER,
        entity::CAMERA_ANSWER_LABEL,
        entity::CAMERA_DET,
        entity::ANSWER_HEADLINE,
        entity::TRACKS_FUSED_FRACTION,
        entity::TRACKS_CAMERA_FUSED_FRACTION,
        entity::LIDAR_SWEEP,
        entity::LIDAR_VOXELS,
        entity::LIDAR_DETECTIONS,
        entity::LIDAR_TRACKS,
        entity::LIDAR_ANSWER,
        entity::LOG_DROPS,
        entity::LOG_EVENTS,
        entity::BYTES_TABLE,
        entity::STAGE_TABLE,
        entity::GRAPH_PIPELINE,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    all.push(format!("{}/line", entity::ANSWER));
    // From the lists the stages log by, so the table cannot name a leaf the
    // stage does not log.
    for leaf in ANSWER_LEAVES {
        all.push(format!("{}/{leaf}", entity::ANSWER));
    }
    for leaf in TRACK_COUNT_LEAVES {
        all.push(format!("{}/{leaf}", entity::TRACKS_COUNT));
    }
    for leaf in PAIRING_LEAVES
        .iter()
        .chain([PAIRING_BAND].iter())
        .chain(READY_DOTS.iter().filter(|d| d.overview).map(|d| &d.leaf))
    {
        all.push(format!("{}/{leaf}", entity::PAIRING));
    }
    for leaf in [SWEEP_END_LEAF]
        .iter()
        .chain(READY_DOTS.iter().map(|d| &d.leaf))
    {
        all.push(format!("{}/{leaf}", entity::AFTER_SWEEP));
    }
    for name in LANE_NAMES {
        all.push(lane_path(name));
    }
    all
}

/// What a stream actually logged into its `kind` of store (the recording, or
/// the blueprint the layout is), read back out of its memory sink: every
/// entity path (without the leading slash), with the components logged on
/// it, static or not, as `Archetype:field`. What the tests check the layout
/// and the styling against, rather than against a list kept by hand beside
/// the code.
#[cfg(test)]
pub fn logged(
    storage: &rerun::sink::MemorySinkStorage,
    kind: rerun::StoreKind,
) -> std::collections::BTreeMap<String, std::collections::BTreeSet<String>> {
    use rerun::log::{Chunk, LogMsg};
    let mut out: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for msg in storage.take() {
        if let LogMsg::ArrowMsg(store, arrow) = msg {
            if store.kind() != kind {
                continue;
            }
            let Ok(chunk) = Chunk::from_arrow_msg(&arrow) else {
                continue;
            };
            let path = chunk.entity_path().to_string();
            let entry = out
                .entry(path.trim_start_matches('/').to_string())
                .or_default();
            for d in chunk.component_descriptors() {
                entry.insert(d.component.to_string());
            }
        }
    }
    out
}

/// Whether the layout is for a viewer watching the run live or for a file
/// opened later. Live, the plots show the last eight seconds and the time
/// panel plays; from a file, the whole run is visible and paused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Live,
    File,
}

/// One view as it was logged: what the test over the layout walks.
#[cfg_attr(not(test), allow(dead_code))] // read by the tests; the run only sends it
#[derive(Clone, Debug)]
pub struct ViewSpec {
    /// `view/<id>`.
    pub path: String,
    /// The viewer's class identifier: `2D`, `3D`, `TimeSeries`, ...
    pub class: String,
    /// The title on the tab.
    pub name: String,
    /// The `+ /path` expressions the view shows.
    pub contents: Vec<String>,
}

/// The layout as it was logged.
#[cfg_attr(not(test), allow(dead_code))] // read by the tests; the run only sends it
#[derive(Debug, Default)]
pub struct Layout {
    pub views: Vec<ViewSpec>,
    /// `container/<id>` for every container, the root last.
    pub containers: Vec<String>,
}

#[cfg(test)]
impl Layout {
    /// Whether some view shows `entity`: a `+ /path` naming it, or a
    /// `+ /path/**` it is at or under.
    pub fn shows(&self, entity: &str) -> bool {
        self.views
            .iter()
            .flat_map(|v| v.contents.iter())
            .filter_map(|c| c.strip_prefix("+ /"))
            .any(|c| match c.strip_suffix("/**") {
                Some(prefix) => entity == prefix || entity.starts_with(&format!("{prefix}/")),
                None => entity == c,
            })
    }
}

/// A UUID-shaped id derived from a name, so the same run twice logs the same
/// blueprint byte for byte. The viewer requires the last part of `view/<id>`
/// and `container/<id>` to parse as a UUID; it does not require randomness.
/// FNV-1a over the seed, twice with different offsets, marked as a
/// version-4/variant-1 UUID so it reads as one.
fn stable_uuid(seed: &str) -> Uuid {
    fn fnv1a(seed: &str, basis: u64) -> u64 {
        seed.bytes().fold(basis, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }
    let hi = fnv1a(seed, 0xcbf2_9ce4_8422_2325);
    let lo = fnv1a(seed, 0x8422_2325_cbf2_9ce4);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&hi.to_be_bytes());
    bytes[8..].copy_from_slice(&lo.to_be_bytes());
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid { bytes }
}

/// The `8-4-4-4-12` text of a UUID, which is how a blueprint path names it.
fn uuid_string(u: &Uuid) -> String {
    let hex: String = u.bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// The pictures' background, camera and lidar alike: a solid #101010, which
/// the teal cloud and the gold answer read against better than a gradient.
fn solid_background() -> Background {
    Background::new(BackgroundKind::SolidColor).with_color(Color::from_rgb(16, 16, 16))
}

/// The pairing plot's y range, ms from the sweep's start: from 0.6 of a
/// period below the band, so a stale frame a whole period early still lands
/// inside the plot, to a little over [`PAIRING_DONE_TOP`] periods above the
/// sweep's start, so a ready dot held there is inside the plot and not on
/// its border. From the run's own period, never from the healthy data;
/// about one 10 Hz period when the run has none.
fn pairing_range(bounds: &Bounds) -> (f64, f64) {
    let p = bounds.period_ms.unwrap_or(FALLBACK_PERIOD_MS);
    (-0.6 * p, (PAIRING_DONE_TOP + 0.1) * p)
}

/// The zoom's y range, ms after the sweep ended: [`SWEEP_END_ZOOM`] and a
/// hundredth of a period more at either end, so a dot held at an edge is
/// inside the plot and not on its border. From the run's own period, like
/// the pairing plot's.
fn sweep_end_range(bounds: &Bounds) -> (f64, f64) {
    let p = bounds.period_ms.unwrap_or(FALLBACK_PERIOD_MS);
    let (lo, hi) = SWEEP_END_ZOOM;
    ((lo - 0.01) * p, (hi + 0.01) * p)
}

/// The period an axis is scaled by when the run could not measure one: the
/// 10 Hz both KITTI sensors run at.
const FALLBACK_PERIOD_MS: f64 = 100.0;

/// The stretch of the `host` timeline every plot shows: live, the last eight
/// seconds up to half a second past the cursor; from a file, the whole run.
fn time_window(mode: Mode) -> TimeRange {
    match mode {
        Mode::Live => TimeRange {
            start: TimeRangeBoundary::CursorRelative(TimeInt(-8_000_000_000)),
            end: TimeRangeBoundary::CursorRelative(TimeInt(500_000_000)),
        },
        Mode::File => TimeRange {
            start: TimeRangeBoundary::Infinite,
            end: TimeRangeBoundary::Infinite,
        },
    }
}

/// The viewer's shared time axis, where every plot's x range comes from.
///
/// A plot whose `TimeAxis` is linked to the global one -- every plot here,
/// so a stack of them reads as one time axis -- takes its x range from the
/// `TimeAxis` of one fixed view id, `re_viewer_context::GLOBAL_VIEW_ID` in
/// rerun 0.38.1 ("a dummy view for shared blueprint data between views"; the
/// SDK does not export it). With no range stored there, each plot fell back
/// to the span of its OWN data, so on a run whose first answer came 3.3 s in,
/// the plots drawn only from answers started at 3.3 s and the ones drawn
/// from every sweep at 0: the time cursor stood at four different places on
/// the Fusion page's four panels. Stored here, it is [`time_window`] for all
/// of them; a pan or zoom in one moves them all, and a double-click returns
/// to it.
pub const GLOBAL_TIME_AXIS: &str = "view/5c0dca6a-e63f-9cf7-f657-2602590474cc/TimeAxis";

/// `+ /path`: one view-contents expression.
fn q(path: &str) -> String {
    format!("+ /{path}")
}

/// How one container lays its children out.
struct Frame<'a> {
    kind: ContainerKind,
    /// The tab title, for a container that is itself a tab.
    name: Option<&'a str>,
    col_shares: &'a [f32],
    row_shares: &'a [f32],
    /// The child shown first, for tabs.
    active: Option<&'a str>,
}

impl<'a> Frame<'a> {
    fn horizontal(col_shares: &'a [f32]) -> Self {
        Frame {
            kind: ContainerKind::Horizontal,
            name: None,
            col_shares,
            row_shares: &[],
            active: None,
        }
    }

    fn vertical(name: Option<&'a str>, row_shares: &'a [f32]) -> Self {
        Frame {
            kind: ContainerKind::Vertical,
            name,
            col_shares: &[],
            row_shares,
            active: None,
        }
    }

    fn tabs(name: Option<&'a str>, active: &'a str) -> Self {
        Frame {
            kind: ContainerKind::Tabs,
            name,
            col_shares: &[],
            row_shares: &[],
            active: Some(active),
        }
    }
}

/// Logs the layout onto a blueprint store, remembering what it logged.
struct Builder<'a> {
    bp: &'a RecordingStream,
    mode: Mode,
    layout: Layout,
}

impl Builder<'_> {
    /// One view: its contents at `view/<id>/ViewContents`, then the view.
    fn view(
        &mut self,
        seed: &str,
        class: &str,
        name: &str,
        origin: &str,
        contents: &[String],
    ) -> Result<String, RecordingStreamError> {
        let path = format!("view/{}", uuid_string(&stable_uuid(seed)));
        self.bp.log(
            format!("{path}/ViewContents"),
            &ViewContents::new(contents.iter().map(|c| QueryExpression::from(c.as_str()))),
        )?;
        self.bp.log(
            path.clone(),
            &ViewBlueprint::new(ViewClass::from(class))
                .with_display_name(name)
                .with_space_origin(origin),
        )?;
        self.layout.views.push(ViewSpec {
            path: path.clone(),
            class: class.to_string(),
            name: name.to_string(),
            contents: contents.to_vec(),
        });
        Ok(path)
    }

    /// One property archetype of a view, at `view/<id>/<Archetype>`: the path
    /// pattern the Python SDK writes and the viewer reads.
    fn prop(
        &self,
        view: &str,
        archetype: &str,
        value: &dyn AsComponents,
    ) -> Result<(), RecordingStreamError> {
        self.bp.log(format!("{view}/{archetype}"), value)
    }

    fn container(
        &mut self,
        seed: &str,
        frame: Frame<'_>,
        children: &[String],
    ) -> Result<String, RecordingStreamError> {
        let path = format!("container/{}", uuid_string(&stable_uuid(seed)));
        let mut arch = ContainerBlueprint::new(frame.kind)
            .with_contents(children.iter().map(|c| IncludedContent::from(c.as_str())));
        if let Some(name) = frame.name {
            arch = arch.with_display_name(name);
        }
        if !frame.col_shares.is_empty() {
            arch = arch.with_col_shares(frame.col_shares.iter().copied());
        }
        if !frame.row_shares.is_empty() {
            arch = arch.with_row_shares(frame.row_shares.iter().copied());
        }
        if let Some(active) = frame.active {
            arch = arch.with_active_tab(ActiveTab::from(active));
        }
        self.bp.log(path.clone(), &arch)?;
        self.layout.containers.push(path.clone());
        Ok(path)
    }

    /// The time window every plot queries: [`time_window`] on the `host`
    /// timeline.
    fn window(&self) -> VisibleTimeRanges {
        VisibleTimeRanges::new([VisibleTimeRange(VisibleTimeRangeSpec {
            timeline: "host".into(),
            range: time_window(self.mode),
        })])
    }

    /// A time-series plot: a y range, locked, from the run's own bounds --
    /// never from the healthy data, which would hide exactly the excursion
    /// an overloaded run makes -- or `None` to fit the data; the global time
    /// axis; no legend, because every plot names its colours in its title;
    /// and the window.
    fn plot(
        &mut self,
        seed: &str,
        name: &str,
        contents: &[String],
        y: Option<(f64, f64)>,
    ) -> Result<String, RecordingStreamError> {
        let v = self.view(seed, "TimeSeries", name, "/", contents)?;
        if let Some((lo, hi)) = y {
            self.prop(
                &v,
                "ScalarAxis",
                &ScalarAxis::new().with_range([lo, hi]).with_zoom_lock(true),
            )?;
        }
        self.prop(
            &v,
            "TimeAxis",
            &TimeAxis::new().with_link(LinkAxis::LinkToGlobal),
        )?;
        self.prop(&v, "PlotLegend", &PlotLegend::new().with_visible(false))?;
        self.prop(&v, "VisibleTimeRanges", &self.window())?;
        Ok(v)
    }

    /// A state-timeline view over `contents`, one lane per entity, over the
    /// whole run live or not: a sticky lane logs a state only when it
    /// changes, so an eight-second window that began after the last change
    /// would show nothing at all. From a file, where the plots show the
    /// whole run too, its time axis is the plots' own, so the Fusion page's
    /// lanes stay over its plot when either is zoomed.
    fn lanes(
        &mut self,
        seed: &str,
        name: &str,
        contents: &[String],
    ) -> Result<String, RecordingStreamError> {
        let v = self.view(seed, "StateTimeline", name, "/", contents)?;
        if self.mode == Mode::File {
            self.prop(
                &v,
                "TimeAxis",
                &TimeAxis::new().with_link(LinkAxis::LinkToGlobal),
            )?;
        }
        self.prop(
            &v,
            "VisibleTimeRanges",
            &VisibleTimeRanges::new([VisibleTimeRange(VisibleTimeRangeSpec {
                timeline: "host".into(),
                range: TimeRange {
                    start: TimeRangeBoundary::Infinite,
                    end: TimeRangeBoundary::Infinite,
                },
            })]),
        )?;
        Ok(v)
    }

    /// Hides `entity` in `view` by default, the viewer's eye toggle for it
    /// turned off: in the view, so the reader can turn it on, but not drawn
    /// until then. Logged as the Python SDK logs an `EntityBehavior`
    /// override, under the view's contents.
    fn hide(&self, view: &str, entity: &str) -> Result<(), RecordingStreamError> {
        self.bp.log(
            format!("{view}/ViewContents/overrides/{entity}/visualizers"),
            &EntityBehavior::new().with_visible(false),
        )
    }

    /// A text log over `contents` showing WARN and ERROR rows only: the
    /// drops and the degraded recorder, never the shutdown's INFO line.
    fn warnings(
        &mut self,
        seed: &str,
        name: &str,
        contents: &[String],
    ) -> Result<String, RecordingStreamError> {
        let v = self.view(seed, "TextLog", name, "/", contents)?;
        self.prop(
            &v,
            "TextLogRows",
            &TextLogRows::new().with_filter_by_log_level([TextLogLevel::WARN, TextLogLevel::ERROR]),
        )?;
        Ok(v)
    }

    /// A 3D view of the lidar on the pictures' solid #101010, from the eye
    /// that fills the view with the scene: 7 m behind the sensor and 15 m
    /// up, looking at a point 22 m down the road and 4 m below it, so the
    /// road runs to the view's top edge with no band of empty sky above it.
    /// A 10 m grid on the ground plane, dim enough to sit behind the teal.
    fn lidar(
        &mut self,
        seed: &str,
        name: &str,
        contents: &[String],
    ) -> Result<String, RecordingStreamError> {
        let v = self.view(seed, "3D", name, "/lidar", contents)?;
        self.prop(&v, "Background", &solid_background())?;
        self.prop(
            &v,
            "LineGrid3D",
            &LineGrid3D::new()
                .with_visible(true)
                .with_spacing(10.0)
                .with_plane(Plane3D::XY)
                .with_stroke_width(1.0)
                .with_color(Color::from_rgb(45, 45, 50)),
        )?;
        self.prop(
            &v,
            "EyeControls3D",
            &EyeControls3D::new()
                .with_kind(Eye3DKind::Orbital)
                .with_position(Position3D::new(-7.0, 0.0, 15.0))
                .with_look_target(Position3D::new(22.0, 0.0, -4.0))
                .with_eye_up([0.0, 0.0, 1.0]),
        )?;
        Ok(v)
    }
}

/// The Pipeline page: what each stage does, how long it took and what it
/// hands on, under the pipeline's shape.
///
/// On top, the pipeline as a graph, each node labelled with the bytes its
/// stage hands on for the latest sweep, and the three nodes the byte chain
/// does not reach -- the camera, the admission and the camera stage -- with
/// a second line saying what they do. Under it the byte chain, the lidar's
/// data per sweep from the sweep to the answer: what each step does, what
/// the link carried, what the step allocated to build it and what the next
/// stage allocated to read it, 0 where it read in place. At the bottom every
/// stage, the camera's and the lidar's, as a table: what it does, its median
/// time per sample and what it hands on.
fn pipeline_page(b: &mut Builder) -> Result<String, RecordingStreamError> {
    let graph = b.view(
        "bytes-graph",
        "Graph",
        "Pipeline - bytes carried per hop (this sweep)",
        &format!("/{}", entity::GRAPH_PIPELINE),
        &[q(entity::GRAPH_PIPELINE)],
    )?;
    // Wide enough for the two-line labels at the left edge, whose second
    // line runs about 80 units either side of a node at x = 0, and tall
    // enough for the rows at +-70.
    b.prop(
        &graph,
        "VisualBounds2D",
        &VisualBounds2D::new(Range2D {
            x_range: Range1D([-100.0, 820.0]),
            y_range: Range1D([-95.0, 95.0]),
        }),
    )?;
    // Every force off: the nodes stay where they are put, the same every
    // sweep, rather than drifting while the reader looks.
    let off = || Enabled(Bool(false));
    b.prop(&graph, "ForceLink", &ForceLink::new().with_enabled(off()))?;
    b.prop(
        &graph,
        "ForceManyBody",
        &ForceManyBody::new().with_enabled(off()),
    )?;
    b.prop(
        &graph,
        "ForcePosition",
        &ForcePosition::new().with_enabled(off()),
    )?;
    b.prop(
        &graph,
        "ForceCollisionRadius",
        &ForceCollisionRadius::new().with_enabled(off()),
    )?;
    b.prop(
        &graph,
        "ForceCenter",
        &ForceCenter::new().with_enabled(off()),
    )?;
    let chain = b.view(
        "bytes-table",
        "TextDocument",
        "Byte chain - the lidar's data per sweep, medians",
        "/",
        &[q(entity::BYTES_TABLE)],
    )?;
    let stages = b.view(
        "stage-table",
        "TextDocument",
        "Stages - what each does and how long it took, medians",
        "/",
        &[q(entity::STAGE_TABLE)],
    )?;
    b.container(
        "bytes",
        Frame::vertical(Some("Pipeline"), &[2.2, 2.7, 3.1]),
        &[graph, chain, stages],
    )
}

/// The camera picture: the frame, every record of the answer in it and every
/// camera-only detection coloured by population, the flagged record in gold
/// with its label, and the stamp saying which frame they were fused with.
/// The detector's raw boxes (`camera/cam_det`) are in the view but hidden
/// (the viewer's eye toggle shows them): every one of them is drawn already,
/// as a fused track's box or a camera-only box, and the raw layer on top
/// turned each object into two overlapping outlines.
fn camera_view(b: &mut Builder, bounds: &Bounds) -> Result<String, RecordingStreamError> {
    let camera = b.view(
        "camera",
        "2D",
        "Camera - tracks by source, answer in gold",
        "/camera",
        &[
            q(entity::CAMERA_IMAGE),
            q(entity::CAMERA_TRACKS),
            q(entity::CAMERA_STATUS),
            q(entity::CAMERA_ANSWER),
            q(entity::CAMERA_ANSWER_LABEL),
            q(entity::CAMERA_DET),
        ],
    )?;
    b.hide(&camera, entity::CAMERA_DET)?;
    b.prop(&camera, "Background", &solid_background())?;
    // The frame's own size as the view's bounds, so the picture fills the
    // view edge to edge from the first frame rather than fitting whatever was
    // drawn first.
    if let Some((w, h)) = bounds.image_wh {
        b.prop(
            &camera,
            "VisualBounds2D",
            &VisualBounds2D::new(Range2D {
                x_range: Range1D([0.0, f64::from(w)]),
                y_range: Range1D([0.0, f64::from(h)]),
            }),
        )?;
    }
    Ok(camera)
}

/// The Demo page: the three plots a first-time reader needs beside the
/// pictures. The time to collision as points against the 3 s line; the
/// pairing, the camera frame's instant as a point inside the sweep's window
/// drawn as a grey band, and on the same axis when the camera's half, the
/// lidar's half and the answer were ready, which says whether the camera
/// frame was ready when the lidar's boxes were; and those finishes zoomed
/// onto the sweep's end, where the healthy run's all fall within a few
/// milliseconds, with `reduce`'s among them. No legends: the titles say what
/// the marks are, and the colours are the sensors' own.
fn demo_page(b: &mut Builder, bounds: &Bounds) -> Result<String, RecordingStreamError> {
    let answer = |leaf: &str| q(&format!("{}/{leaf}", entity::ANSWER));
    let ttc = b.plot(
        "demo-ttc",
        "Time to collision (s) - grey: 3 s",
        &[answer("ttc_s"), answer("ttc_warn")],
        Some((0.0, TTC_TOP_S)),
    )?;
    let pairing = b.plot(
        "demo-pairing",
        "Pairing (ms from sweep start) - photo (dot); done: camera, detect, answer (diamonds)",
        &[q(&format!("{}/**", entity::PAIRING))],
        Some(pairing_range(bounds)),
    )?;
    let zoom = b.plot(
        "demo-sweep-end",
        "Sweep's end, zoomed (ms after it) - done: camera, reduce, detect, answer",
        &[q(&format!("{}/**", entity::AFTER_SWEEP))],
        Some(sweep_end_range(bounds)),
    )?;
    b.container(
        "demo",
        Frame::vertical(Some("Demo"), &[2.6, 3.2, 2.6]),
        &[ttc, pairing, zoom],
    )
}

/// The Fusion page: whether the pipeline kept up, and what the fusion made
/// of it, on one time axis. On top the five sticky health lanes over the
/// whole run -- the pairing (stale or expired, the consequence at the
/// fusion; grey where a sensor's source had no frame) over the queues in
/// front of it (dropping or skipping) -- so a pairing that failed sits over
/// the queue that failed it. Under them the tracks in the camera frame by
/// population, and the fused share of each sensor's objects in the frame:
/// of the lidar's tracks, the ones the camera confirmed, and of the
/// camera's detections, the ones the lidar did -- what the answer lost.
fn fusion_page(b: &mut Builder) -> Result<String, RecordingStreamError> {
    // Five lanes are 40 px each (a 14 px label, a 22 px band, a gap) under
    // a 20 px time axis and the title bar: 244 px, which 2.7 of 7.1 more
    // than gives at 1600x900.
    let health = b.lanes(
        "fusion-outcome",
        "Health - green ok, amber stale or skipping, red expired or dropping, grey gap",
        &LANE_NAMES.map(|n| q(&lane_path(n))),
    )?;
    // Fitted to the data: the populations' sizes differ by a factor of ten,
    // and no bound the run sets says how many objects a street holds.
    let counts = b.plot(
        "fusion-counts",
        "Tracks in frame by source - fused (green), lidar-only (teal), camera-only (blue)",
        &[q(&format!("{}/**", entity::TRACKS_COUNT))],
        None,
    )?;
    let share = b.plot(
        "fusion-share",
        "Fused share - of the lidar tracks in frame (teal), of the camera's boxes (blue)",
        &[
            q(entity::TRACKS_FUSED_FRACTION),
            q(entity::TRACKS_CAMERA_FUSED_FRACTION),
        ],
        Some((0.0, 1.0)),
    )?;
    b.container(
        "fusion",
        Frame::vertical(Some("Fusion"), &[2.7, 2.2, 2.2]),
        &[health, counts, share],
    )
}

/// The Log page: the WARN and ERROR rows -- every drop with its reason, and
/// a degraded recorder -- over the answer as one sentence per sweep, the
/// auditor's record of what the chain said.
fn log_page(b: &mut Builder) -> Result<String, RecordingStreamError> {
    let events = b.warnings("log-events", "Events (WARN+)", &[q("log/**")])?;
    let line = b.view(
        "answer-line",
        "TextLog",
        "Answer - one line per sweep (auditor)",
        "/answer",
        &[q(&format!("{}/line", entity::ANSWER))],
    )?;
    // Body only: the four timeline columns and the entity path say nothing a
    // reader of this log needs, and they pushed the sentence off the panel.
    let hidden = |timeline: &str| {
        TimelineColumn(TimelineColumnSpec {
            visible: Bool(false),
            timeline: timeline.into(),
        })
    };
    let text_col = |kind: TextLogColumnKind, visible: bool| {
        TextLogColumn(TextLogColumnSpec {
            visible: Bool(visible),
            kind,
        })
    };
    b.prop(
        &line,
        "TextLogColumns",
        &TextLogColumns::new()
            .with_timeline_columns([
                hidden("host"),
                hidden("sensor_time"),
                hidden("seq"),
                hidden("log_time"),
            ])
            .with_text_log_columns([
                text_col(TextLogColumnKind::Body, true),
                text_col(TextLogColumnKind::EntityPath, false),
                text_col(TextLogColumnKind::LogLevel, false),
            ]),
    )?;
    b.container(
        "log",
        Frame::vertical(Some("Log"), &[1.0, 2.0]),
        &[events, line],
    )
}

/// The layout, logged onto `bp`.
///
/// Left column (13 of 21), what the chain says: the answer as a headline in
/// large type; the camera picture, edge to edge; and under it the lidar,
/// the voxels with the track wireframes and the gold answer, with the raw
/// sweep on a second tab. Right column (8 of 21), four tabs: Demo, the three
/// plots a first-time reader needs, shown first; then the auditor's pages,
/// one full-height page each -- Fusion (the health lanes and what the fusion
/// confirmed), Pipeline (each stage's work, time and bytes), and Log.
fn build(
    bp: &RecordingStream,
    mode: Mode,
    bounds: &Bounds,
) -> Result<Layout, RecordingStreamError> {
    let mut b = Builder {
        bp,
        mode,
        layout: Layout::default(),
    };

    // ---- left column ----------------------------------------------------
    let headline = b.view(
        "headline",
        "TextDocument",
        "Answer - nearest object in the corridor",
        "/",
        &[q(entity::ANSWER_HEADLINE)],
    )?;
    let camera = camera_view(&mut b, bounds)?;
    let lidar = b.lidar(
        "lidar",
        "Lidar - voxels, tracks",
        &[
            q(entity::LIDAR_VOXELS),
            q(entity::LIDAR_TRACKS),
            q(entity::LIDAR_ANSWER),
        ],
    )?;
    // The detections a track was built from are in the raw view but hidden:
    // at a sweep's density their boxes and the tracks' coincide.
    let lidar_raw = b.lidar(
        "lidar-raw",
        "Raw sweep",
        &[
            q(entity::LIDAR_SWEEP),
            q(entity::LIDAR_TRACKS),
            q(entity::LIDAR_ANSWER),
            q(entity::LIDAR_DETECTIONS),
        ],
    )?;
    b.hide(&lidar_raw, entity::LIDAR_DETECTIONS)?;
    let lidar_tabs = b.container(
        "lidar-tabs",
        Frame::tabs(None, &lidar),
        &[lidar.clone(), lidar_raw],
    )?;
    // The headline's share is the height that shows one `##` heading line
    // without clipping its descenders at 1600x900.
    let left = b.container(
        "left",
        Frame::vertical(None, &[0.9, 3.35, 4.45]),
        &[headline, camera, lidar_tabs],
    )?;

    // ---- right column: the demo and the auditor's pages -------------------
    let demo = demo_page(&mut b, bounds)?;
    let fusion = fusion_page(&mut b)?;
    let pipeline = pipeline_page(&mut b)?;
    let log = log_page(&mut b)?;
    let right = b.container(
        "right",
        Frame::tabs(None, &demo),
        &[demo.clone(), fusion, pipeline, log],
    )?;
    let root = b.container("root", Frame::horizontal(&[13.0, 8.0]), &[left, right])?;

    // ---- the viewport and the panels --------------------------------------
    bp.log(
        "viewport",
        &ViewportBlueprint::new()
            .with_root_container(RootContainer(stable_uuid("root")))
            .with_auto_layout(AutoLayout(Bool(false)))
            .with_auto_views(AutoViews(Bool(false))),
    )?;
    debug_assert!(root.ends_with(&uuid_string(&stable_uuid("root"))));
    // The one time axis every plot is linked to.
    bp.log(
        GLOBAL_TIME_AXIS,
        &TimeAxis::new().with_view_range(time_window(mode)),
    )?;
    for panel in ["blueprint_panel", "selection_panel", "top_panel"] {
        bp.log(
            panel,
            &PanelBlueprint::new().with_state(PanelState::Collapsed),
        )?;
    }
    bp.log(
        "time_panel",
        &TimePanelBlueprint::new()
            .with_state(PanelState::Collapsed)
            .with_timeline("host")
            .with_playback_speed(1.0)
            .with_loop_mode(LoopMode::Off)
            .with_play_state(match mode {
                Mode::Live => PlayState::Playing,
                Mode::File => PlayState::Paused,
            }),
    )?;
    Ok(b.layout)
}

/// Builds the layout as a blueprint store and sends it on `rec`, active and
/// default, so it precedes the first data row: a viewer the run spawned
/// opens straight into it, and an `.rrd` file carries it at its head.
///
/// A stream that is not enabled (`--rerun off`) has no store to send to, and
/// sends nothing.
pub fn send_blueprint(
    rec: &RecordingStream,
    mode: Mode,
    bounds: &Bounds,
) -> Result<Layout, RecordingStreamError> {
    let (bp, storage) = RecordingStreamBuilder::new("pipes")
        .recording_id(RecordingId::random())
        .blueprint()
        .memory()?;
    // Required for the viewer to identify blueprint data.
    bp.set_time_sequence("blueprint", 0);
    let layout = build(&bp, mode, bounds)?;
    if let Some(info) = bp.store_info() {
        rec.send_blueprint(
            storage.take(),
            BlueprintActivationCommand {
                blueprint_id: info.store_id,
                make_active: true,
                make_default: true,
            },
        );
    }
    Ok(layout)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn every_edge_is_named_once_and_nothing_hangs_under_pipes() {
        let mut seen: Vec<&str> = EDGES.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), EDGES.len(), "an edge is listed twice");
        for e in EDGES {
            assert!(e.contains("->"), "{e} is not a queue");
        }
        // The two sensor drivers and nothing else: a derived stream's producer
        // row cannot be Missing, so it can never say a frame is absent.
        assert_eq!(ADMISSION_EDGES, ["cam0", "velo"]);
        for e in ADMISSION_EDGES {
            assert!(!e.contains("->"), "{e} is a queue, not a driver");
            assert!(!EDGES.contains(&e));
        }
        let all = entities();
        let mut sorted = all.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), all.len(), "an entity is listed twice");
        for p in &all {
            assert!(!p.starts_with('/'), "{p} has a leading slash");
            assert!(
                !p.starts_with("pipes"),
                "{p} is filed by thread, not by meaning"
            );
        }
    }

    #[test]
    fn the_chart_and_table_edges_are_real_edges_in_chain_order() {
        // The byte chain is read where the next stage of the chain reads it,
        // and built where the step before made it: the sweep before the
        // voxels, the voxels before the detections, and so on.
        let pos = |e: &str| EDGES.iter().position(|x| *x == e).unwrap();
        for link in CHAIN_BYTES {
            assert!(
                EDGES.contains(&link.edge),
                "{}: {} is not a queue this binary opens",
                link.name,
                link.edge
            );
            assert!(
                !link.built_by.contains("->"),
                "{}: {} is a queue, not the row of the step that built it",
                link.name,
                link.built_by
            );
            assert!(!link.step.is_empty(), "{} says nothing", link.name);
        }
        for w in CHAIN_BYTES.windows(2) {
            assert!(
                pos(w[0].edge) < pos(w[1].edge),
                "{} is read after {}",
                w[0].name,
                w[1].name
            );
        }
        // The lidar driver's own row built the sweep: the chain's one copy.
        assert_eq!(CHAIN_BYTES[0].built_by, "velo");
        assert!(ADMISSION_EDGES.contains(&CHAIN_BYTES[0].built_by));
        // The ready dots: the camera's half, the lidar's two stages --
        // `detect` done is the lidar's half -- and the answer, the chain's
        // end, each timed on a real edge. The pairing plot leaves `reduce`
        // to the zoom.
        assert_eq!(
            READY_DOTS.map(|d| (d.clock, d.overview)),
            [
                (StageClock::Camera, true),
                (StageClock::Edge("velo->reduce"), false),
                (StageClock::Edge("det->detect"), true),
                (StageClock::Edge("state->sink"), true),
            ]
        );
        let edges: Vec<&str> = READY_DOTS
            .iter()
            .filter_map(|d| match d.clock {
                StageClock::Edge(e) => Some(e),
                StageClock::Camera => None,
            })
            .collect();
        for e in &edges {
            assert!(EDGES.contains(e), "{e} is not a queue this binary opens");
        }
        for w in edges.windows(2) {
            assert!(pos(w[0]) < pos(w[1]), "{} is drawn after {}", w[0], w[1]);
        }
    }

    #[test]
    fn every_ready_dot_is_numbered_in_the_order_it_is_drawn() {
        // The viewer draws a plot's series in entity order, the last on top:
        // each leaf is its place, from 1, then what it is, and none is a
        // leaf the pairing already has.
        for (i, d) in READY_DOTS.iter().enumerate() {
            let n = d.leaf;
            let (num, word) = n.split_once('_').unwrap();
            assert_eq!(num, (i + 1).to_string(), "{n}");
            assert!(
                word.ends_with("_done")
                    && word
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "{n:?}"
            );
            assert!(!PAIRING_LEAVES.contains(&n) && n != PAIRING_BAND, "{n}");
        }
        // Every one is in the entity table under the zoom, and the pairing
        // plot's under the pairing too; each plot shows its root.
        let all = entities();
        for d in READY_DOTS {
            assert!(all.contains(&format!("{}/{}", entity::AFTER_SWEEP, d.leaf)));
            assert_eq!(
                all.contains(&format!("{}/{}", entity::PAIRING, d.leaf)),
                d.overview,
                "{}",
                d.leaf
            );
        }
        // A dot held at an edge is inside its plot: over the band on the
        // pairing plot, and either side of the sweep's end on the zoom.
        let (lo, hi) = pairing_range(&bounds());
        assert!(lo < 0.0 && PAIRING_DONE_TOP * 103.3 < hi, "{lo} {hi}");
        let (lo, hi) = sweep_end_range(&bounds());
        let (early, late) = SWEEP_END_ZOOM;
        assert!(lo < early * 103.3 && early < 0.0 && 0.0 < late && late * 103.3 < hi);
        // The zoom's line is the pairing plot's `sweep_end`, so it is styled
        // as the same reference.
        assert!(PAIRING_LEAVES.contains(&SWEEP_END_LEAF));
    }

    #[test]
    fn the_pipeline_graph_joins_every_node_it_draws() {
        let ids: Vec<&str> = GRAPH_NODES.iter().map(|(id, _, _)| *id).collect();
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "a node twice");
        for (from, to) in GRAPH_EDGES {
            assert!(ids.contains(&from) && ids.contains(&to), "{from} -> {to}");
        }
        // Every node is on some edge, and the answer is the one with none
        // leaving it.
        for id in &ids {
            assert!(
                GRAPH_EDGES.iter().any(|(f, t)| f == id || t == id),
                "{id} is on no edge"
            );
        }
        assert!(!GRAPH_EDGES.iter().any(|(f, _)| *f == "answer"));
        // Left to right is upstream to downstream.
        let x = |id: &str| GRAPH_NODES.iter().find(|n| n.0 == id).unwrap().1;
        for (from, to) in GRAPH_EDGES {
            assert!(x(from) < x(to), "{from} -> {to} runs backwards");
        }
    }

    #[test]
    fn a_view_id_is_a_uuid_and_the_same_one_every_time() {
        let a = uuid_string(&stable_uuid("camera"));
        assert_eq!(a, uuid_string(&stable_uuid("camera")));
        assert_ne!(a, uuid_string(&stable_uuid("lidar")));
        // 8-4-4-4-12, lower-case hex, version 4, variant 1.
        let parts: Vec<&str> = a.split('-').collect();
        assert_eq!(
            parts.iter().map(|p| p.len()).collect::<Vec<_>>(),
            [8, 4, 4, 4, 12]
        );
        assert!(a.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        assert!(parts[2].starts_with('4'), "{a}");
        assert!(
            matches!(parts[3].chars().next(), Some('8' | '9' | 'a' | 'b')),
            "{a}"
        );
    }

    fn bounds() -> Bounds {
        Bounds {
            image_wh: Some((1242.0, 375.0)),
            lidar_lane: true,
            answer: true,
            period_ms: Some(103.3),
            camera_edge: Some("cam0->camdet"),
            camera_delay_ms: 0,
            rate: Some(1.0),
        }
    }

    fn layout(mode: Mode) -> Layout {
        let (bp, _storage) = RecordingStreamBuilder::new("pipes-test")
            .blueprint()
            .memory()
            .unwrap();
        bp.set_time_sequence("blueprint", 0);
        build(&bp, mode, &bounds()).unwrap()
    }

    #[test]
    fn dashboard_paths_exist() {
        // Every path a view shows is an entity the run logs, or a prefix of
        // one: the tree is the layout, and a view of nothing is a blank tab.
        let all = entities();
        let l = layout(Mode::File);
        assert!(!l.views.is_empty());
        for v in &l.views {
            for c in &v.contents {
                let path = c
                    .strip_prefix("+ /")
                    .unwrap_or_else(|| panic!("{}: {c:?} is not a `+ /path` expression", v.name));
                let prefix = path.strip_suffix("/**").unwrap_or(path);
                assert!(
                    all.iter()
                        .any(|e| e == prefix || e.starts_with(&format!("{prefix}/"))),
                    "{}: {c} shows nothing the run logs",
                    v.name
                );
            }
        }
    }

    #[test]
    fn every_logged_entity_is_in_a_view() {
        // The other half of `dashboard_paths_exist`: every entity in the
        // table is shown by some view, directly or under a `/**`. v1 logged
        // 31 entities no view showed; a series nobody can see is a cost with
        // no reader. The two annotation contexts are the exception: they are
        // not drawn, they name the classes of what is. The table is kept by
        // hand; what a run actually logs is checked against the layout by
        // `record::tests::every_series_the_dashboard_logs_is_styled_and_in_a_view`
        // (the recorder's mirror) and `tests/dashboard.rs` (a whole run).
        let l = layout(Mode::File);
        for e in entities() {
            if e == entity::CAMERA_ROOT || e == entity::LIDAR_ROOT {
                continue;
            }
            assert!(l.shows(&e), "{e} is logged but no view shows it");
        }
    }

    #[test]
    fn the_layout_is_the_spec_s_and_deterministic() {
        let l = layout(Mode::Live);
        let names: Vec<&str> = l.views.iter().map(|v| v.name.as_str()).collect();
        // The views of the first screen, and a view from each auditor page.
        for want in [
            "Answer - nearest object in the corridor",
            "Camera - tracks by source, answer in gold",
            "Lidar - voxels, tracks",
            "Raw sweep",
            "Time to collision (s) - grey: 3 s",
            "Pairing (ms from sweep start) - photo (dot); done: camera, detect, answer (diamonds)",
            "Sweep's end, zoomed (ms after it) - done: camera, reduce, detect, answer",
            "Tracks in frame by source - fused (green), lidar-only (teal), camera-only (blue)",
            "Health - green ok, amber stale or skipping, red expired or dropping, grey gap",
            "Fused share - of the lidar tracks in frame (teal), of the camera's boxes (blue)",
            "Pipeline - bytes carried per hop (this sweep)",
            "Byte chain - the lidar's data per sweep, medians",
            "Stages - what each does and how long it took, medians",
            "Answer - one line per sweep (auditor)",
        ] {
            assert!(names.contains(&want), "no view named {want:?} in {names:?}");
        }
        // The pages that were cut stay cut: no queue plots, no ratios over a
        // bound, no bar charts and no storage lane.
        for gone in [
            "Fill: depth at push / cap",
            "Age / bound (1 = at bound)",
            "Camera service / period",
            "Alloc / carried (0 = zero-copy)",
            "Storage shared",
        ] {
            assert!(!names.contains(&gone), "{gone:?} is back");
        }
        assert!(!l.views.iter().any(|v| v.class == "BarChart"));
        // The first screen is six views: the headline, the camera, the
        // lidar, and the Demo tab's three plots, which is the right column's
        // first tab.
        let demo: Vec<&ViewSpec> = l
            .views
            .iter()
            .filter(|v| v.path.contains(&uuid_string(&stable_uuid("demo-ttc"))))
            .collect();
        assert_eq!(demo.len(), 1);
        // No tab is named after an entity path, and every class is one the
        // viewer has.
        let all = entities();
        for v in &l.views {
            assert!(
                !v.name.contains("_to_")
                    && !all
                        .iter()
                        .any(|e| v.name == *e || v.name.starts_with(&format!("{e}/"))),
                "{:?} is named after a path",
                v.name
            );
            assert!(
                matches!(
                    v.class.as_str(),
                    "2D" | "3D"
                        | "TimeSeries"
                        | "TextLog"
                        | "BarChart"
                        | "StateTimeline"
                        | "TextDocument"
                        | "Graph"
                ),
                "{:?} has class {:?}",
                v.name,
                v.class
            );
        }
        // Every id is distinct, and the same on a second build.
        let mut paths: Vec<&str> = l
            .views
            .iter()
            .map(|v| v.path.as_str())
            .chain(l.containers.iter().map(String::as_str))
            .collect();
        let n = paths.len();
        paths.sort_unstable();
        paths.dedup();
        assert_eq!(paths.len(), n, "two views or containers share an id");
        let again = layout(Mode::Live);
        assert_eq!(
            l.views.iter().map(|v| &v.path).collect::<Vec<_>>(),
            again.views.iter().map(|v| &v.path).collect::<Vec<_>>()
        );
        assert_eq!(l.containers, again.containers);
        // The root is logged last, so the viewport can name it.
        assert!(l
            .containers
            .last()
            .unwrap()
            .ends_with(&uuid_string(&stable_uuid("root"))));
    }

    #[test]
    fn the_pairing_plot_and_the_health_lanes_are_one_view_each() {
        let l = layout(Mode::File);
        // One pairing plot, the Demo page's, with the ready dots under the
        // same root as the window and the frame; and one zoom on the sweep's
        // end, with its own root.
        for root in [entity::PAIRING, entity::AFTER_SWEEP] {
            let plots: Vec<&ViewSpec> = l
                .views
                .iter()
                .filter(|v| {
                    v.contents
                        .iter()
                        .any(|c| c.starts_with(&format!("+ /{root}")))
                })
                .collect();
            assert_eq!(plots.len(), 1, "{root}: {plots:?}");
            assert_eq!(plots[0].contents, [format!("+ /{root}/**")]);
            assert_eq!(plots[0].class, "TimeSeries");
        }
        // One share plot, both sensors' shares on it.
        let shares: Vec<&ViewSpec> = l
            .views
            .iter()
            .filter(|v| v.contents.iter().any(|c| c.starts_with("+ /tracks/")))
            .filter(|v| !v.contents.iter().any(|c| c.contains("count")))
            .collect();
        assert_eq!(shares.len(), 1, "{shares:?}");
        assert_eq!(
            shares[0].contents,
            [
                q(entity::TRACKS_FUSED_FRACTION),
                q(entity::TRACKS_CAMERA_FUSED_FRACTION)
            ]
        );
        // One lane view, the Fusion page's, with every lane in chain order:
        // the pairing on top, over the queues in front of it.
        let lanes: Vec<&ViewSpec> = l
            .views
            .iter()
            .filter(|v| v.class == "StateTimeline")
            .collect();
        assert_eq!(lanes.len(), 1, "{lanes:?}");
        assert_eq!(
            lanes[0].contents,
            LANE_NAMES.map(|n| q(&lane_path(n))).to_vec()
        );
        assert_eq!(LANE_NAMES[0], LANE_PAIRING_NAME);
    }

    #[test]
    fn every_plot_reads_one_shared_time_axis_set_to_the_window() {
        // The viewer's own id for the view that holds what views share
        // (`re_viewer_context::GLOBAL_VIEW_ID`, rerun 0.38.1).
        let global = Uuid {
            bytes: [
                0x5C, 0x0D, 0xCA, 0x6A, 0xE6, 0x3F, 0x9C, 0xF7, 0xF6, 0x57, 0x26, 0x02, 0x59, 0x04,
                0x74, 0xCC,
            ],
        };
        assert_eq!(
            GLOBAL_TIME_AXIS,
            format!("view/{}/TimeAxis", uuid_string(&global))
        );
        // The window: the whole run from a file, the last eight seconds live.
        let file = time_window(Mode::File);
        assert_eq!(file.start, TimeRangeBoundary::Infinite);
        assert_eq!(file.end, TimeRangeBoundary::Infinite);
        let live = time_window(Mode::Live);
        assert_eq!(
            live.start,
            TimeRangeBoundary::CursorRelative(TimeInt(-8_000_000_000))
        );
        assert_eq!(
            live.end,
            TimeRangeBoundary::CursorRelative(TimeInt(500_000_000))
        );
        for mode in [Mode::File, Mode::Live] {
            let (bp, storage) = RecordingStreamBuilder::new("pipes-test")
                .blueprint()
                .memory()
                .unwrap();
            bp.set_time_sequence("blueprint", 0);
            let l = build(&bp, mode, &bounds()).unwrap();
            bp.flush_blocking().unwrap();
            let logged = logged(&storage, rerun::StoreKind::Blueprint);
            // The shared range is stored, on the viewer's global view.
            let shared = logged
                .get(GLOBAL_TIME_AXIS)
                .unwrap_or_else(|| panic!("{mode:?}: no shared time axis"));
            assert!(
                shared.iter().any(|c| c.contains("view_range")),
                "{mode:?}: {shared:?}"
            );
            // And every plot is linked to it -- from a file, the lanes too.
            for v in &l.views {
                let linked = logged
                    .get(&format!("{}/TimeAxis", v.path))
                    .is_some_and(|c| c.iter().any(|c| c.contains("link")));
                let want =
                    v.class == "TimeSeries" || (v.class == "StateTimeline" && mode == Mode::File);
                assert_eq!(linked, want, "{mode:?}: {} ({})", v.name, v.class);
            }
        }
    }
}
