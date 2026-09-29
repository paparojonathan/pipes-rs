//! `StreamId` and the `Sample` envelope every stage passes around (design §4).
//! Consumers receive `Arc<Sample>`; the payload is never mutated after admit.

use arrow::array::{Array, RecordBatch};
use serde::{Deserialize, Serialize};

use crate::clock::{HostTime, Tov};

/// Identifies a stream: cam0=0, lidar=1, oxts=2, lidar_det=3, cam_det=4,
/// ego=5, tracks=6, lidar_obj=7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StreamId(pub u8);

impl StreamId {
    /// Left colour camera (`image_02`) — the only stream this sprint replays.
    pub const CAM0: StreamId = StreamId(0);
    /// Lidar point cloud (phase 2).
    pub const LIDAR: StreamId = StreamId(1);
    /// OXTS GPS/IMU rows (phase 2).
    pub const OXTS: StreamId = StreamId(2);
    /// A stream **derived from** a lidar scan by a pipeline stage rather than
    /// read off a sensor: the voxel-reduced cloud `reduce` produces.
    ///
    /// The name says "det" and the payload is a point cloud, which is a
    /// mismatch this id has carried since before there were detections. It is
    /// kept rather than corrected: every recorded run under `runs/` writes
    /// this number into `evidence.csv`, and renaming it would silently change
    /// what those files mean. The detections themselves are
    /// [`StreamId::LIDAR_OBJ`].
    ///
    /// A derived sample carries [`Sample::parent`], and it inherits the
    /// sensor's [`Sample::seq`] -- the frame's number -- as well as its
    /// [`Sample::tov`] and [`Sample::due`]: its time of validity is when the
    /// world was measured and never when a stage got round to transforming
    /// it, and one frame keeps one number on every edge.
    pub const LIDAR_DET: StreamId = StreamId(3);
    /// The camera's half of the fusion: one sample per frame a camera stage
    /// **finished**, carrying that frame's instant and bounds and -- with the
    /// frozen detector on -- what was found in it.
    ///
    /// Two payloads, one producer at a time, and the format string says which:
    ///
    /// - `--detector on` (the default): the `camdet` stage's batch,
    ///   `cam_det_yolox_nano_v1` -- the frame reference plus one row per
    ///   detection (box, class id, class name, confidence), with the model's
    ///   name and sha256 in the schema metadata. About 490 B a frame on
    ///   drive_0005, 8 detections on average.
    /// - `--detector off`: `proc`'s bare reference, `cam_frame_ref_v1`, which is
    ///   deliberately not a detection format. Its 152 B hold `frame_seq` and
    ///   `tov_ns` (i64), `width` and `height` (u32) -- 24 B -- and the 16-byte
    ///   format string with its two i32 offsets, whose two buffers arrow's
    ///   string builder rounds up to 64 B of capacity each, which
    ///   [`payload_bytes`] counts. 48 B of content, and nothing found in the
    ///   picture.
    ///
    /// The name says "det" and was declared for the camera's contribution to
    /// a fusion long before either payload existed; under `--detector off` it
    /// carries no detection, which is the mismatch [`StreamId::LIDAR_DET`]
    /// also has, kept for the same kind of reason. `track` reads the frame
    /// reference by format rather than by stream id, and the same four
    /// columns off either payload; off the detector's it also reads the
    /// detections, and associates them with its tracks
    /// (`pipes_kitti::fuse`).
    ///
    /// **Why it is produced by a consumer rather than by the driver.** The
    /// camera half of a pair has to be *downstream of a camera stage*, or the
    /// experiment this project exists to run cannot reach it: a fusion taking
    /// its camera input straight off `Admission` would pair 154 of 154 frames
    /// while the camera stage dropped half of them. Taking it from the stage
    /// makes a camera drop propagate into the fusion for free, and the CAUSE
    /// is a drop row that already exists. Under `--detector off` that stage is
    /// `proc`; under `--detector on` it is `camdet`, on its own queue.
    /// `--consumer-delay-ms` and `--cap` reach whichever of the two it is, so
    /// slowing the camera on purpose reaches the fusion either way; with the
    /// detector the network's own service time can make the camera late too.
    pub const CAM_DET: StreamId = StreamId(4);
    /// The answer: what the whole chain exists to produce, one sample per
    /// sweep that reached the end of it.
    ///
    /// One record per tracked object — its distance, closing speed, time to
    /// contact, age, where it is in the camera image or that it is not in
    /// it, and the class the camera gave it when the fusion associated it
    /// with a detection — with the nearest one in the vehicle's path flagged,
    /// plus the camera's detections that matched no track: about 12.9 kB
    /// against the 1,947,870 B of laser returns the same sweep started as.
    /// Named `ego` because it is a statement about this vehicle's situation
    /// rather than about any one object.
    pub const EGO: StreamId = StreamId(5);
    /// Tracks: detections associated across sweeps, each with a stable id, an
    /// age and a velocity -- and, where the fusion associated it with a camera
    /// detection of the paired frame, that detection's class.
    ///
    /// The first stream in this project that knows something **no single
    /// sample contains**. Every id before it was a restatement of one sweep at
    /// lower resolution; a track's velocity exists only because two sweeps
    /// were compared, so this is where the chain stops compressing and starts
    /// inferring.
    pub const TRACKS: StreamId = StreamId(6);
    /// Objects the `detect` stage found in a reduced cloud: ground removed,
    /// the rest grouped into connected structures.
    ///
    /// Its own id and **not** [`StreamId::LIDAR_DET`], although that name
    /// would fit it better, because that id is already the voxel cloud in
    /// every recorded run on disk. Two streams under one id would also break
    /// `Evidence::driver_admitted`, which is keyed on `(stage, seq)`.
    ///
    /// The first derived stream in this project whose payload is **not** the
    /// shape it consumed: a detection is not a point, so `velo_xyzr` does not
    /// read it and must not.
    pub const LIDAR_OBJ: StreamId = StreamId(7);

    /// Human-readable name; `"unknown"` for ids outside the table.
    pub fn name(self) -> &'static str {
        match self {
            StreamId::CAM0 => "cam0",
            StreamId::LIDAR => "lidar",
            StreamId::OXTS => "oxts",
            StreamId::LIDAR_DET => "lidar_det",
            StreamId::CAM_DET => "cam_det",
            StreamId::EGO => "ego",
            StreamId::TRACKS => "tracks",
            StreamId::LIDAR_OBJ => "lidar_obj",
            _ => "unknown",
        }
    }
}

/// One sample of one stream: the header every stage stamps plus the Arrow payload.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Which stream produced this sample.
    pub stream: StreamId,
    /// The frame number: the one KITTI names the sample's file by, which the
    /// driver gives it and every sample derived from it keeps, so a frame has
    /// one number on every stream, edge and stage. Unique per stream, since a
    /// stage makes at most one output per input.
    pub seq: u64,
    /// Global admission order; 0 until Admission assigns it (M4).
    pub arrival_seq: u64,
    /// The parent whose [`Sample::tov`] and [`Sample::due`] this sample
    /// inherits — **not** every input that went into producing it.
    ///
    /// It said "provenance of derived samples", and that became false the
    /// moment a stage had two inputs. `track` fuses a sweep's detections with
    /// a camera frame; only one of those can be named here, and the one named
    /// is the one the envelope's times come from, because a `tov` inherited
    /// from A while `parent` points at B is two disagreeing claims in adjacent
    /// columns. The camera half travels in the payload instead (`cam_seq`,
    /// `cam_tov_ns`, `pair_age_ns`), which is the same place `tov_trigger_ns`
    /// travels and for the same reason — the stage that needs it is reading
    /// the payload anyway.
    ///
    /// **The debt that leaves:** `Option<(StreamId, u64)>` is the wrong
    /// schema for a fusion and `Vec<(StreamId, u64)>` is the right one. It was
    /// not changed here because `None` and `vec![]` would become two spellings
    /// of "no parent" — the exact ambiguity [`crate::evidence::Evidence::arrival_seq`]
    /// was changed away from — and because it would put a heap allocation on
    /// every derived sample in a project that counts them. The cost of not
    /// changing it is that `run::parent_check` covers half the provenance, so
    /// the other half is checked by `run::pair_check` against these payload
    /// columns instead. A second parent written and never joined is exactly
    /// the failure this field already had once: it was set to `None` at every
    /// site, read nowhere, and a derived sample naming the WRONG sweep passed
    /// every test in the workspace.
    pub parent: Option<(StreamId, u64)>,
    /// Time of validity on the sensor clock — when the measurement was taken,
    /// never when it was processed.
    pub tov: Tov,
    /// `ClockModel` epoch.
    pub epoch: u32,
    /// `None` at `rate_factor = inf`.
    pub due: Option<HostTime>,
    /// When the driver finished producing it; `arrival - due` is pacing error.
    pub arrival: HostTime,
    /// Arc-backed columns; never mutated after admit.
    pub payload: RecordBatch,
    /// `Buffer::as_ptr` of the pixel buffer, set by the driver.
    pub storage_id: usize,
    /// Driver decode cost, reported separately from pipeline cost.
    pub decode_ns: i64,
}

impl Sample {
    /// Bytes of Arrow data this sample carries; see [`payload_bytes`].
    pub fn payload_bytes(&self) -> usize {
        payload_bytes(&self.payload)
    }
}

/// Size in bytes of the Arrow data a batch carries - "how big is this
/// transfer", the number that belongs on a dashboard edge.
///
/// **Counts** every Arrow buffer reachable from the batch's columns: values,
/// offsets and validity bitmaps, and the same for nested children, so a lidar
/// sweep's `LargeList<FixedSizeList<f32, 4>>` reports its point buffer and not
/// just its two offsets. It is the sum of [`Array::get_buffer_memory_size`]
/// over the columns.
///
/// **Excludes** everything that is not that data: the schema (field names,
/// `DataType`s, metadata), the Rust-side structures that point at the buffers
/// (`RecordBatch`, the `ArrayRef`s and their `Arc` control blocks - that is
/// `RecordBatch::get_array_memory_size`, which grows with the number of columns
/// rather than with the data), and the [`Sample`] header itself.
///
/// **It is not [`crate::alloc::bytes_alloc`]**, and the difference is the
/// project's central claim rather than a detail. `bytes_alloc` is what a stage
/// REQUESTED FROM THE ALLOCATOR; this is what the transfer CARRIES. A zero-copy
/// hand-off of a lidar sweep moves 1,974,352 B across the edge while the
/// consumer allocates approximately nothing, and the two numbers shown side by
/// side are what makes that visible. Nor is it [`Sample::storage_id`], which is
/// one buffer's ADDRESS - equal at every stage, and no size at all.
///
/// Two caveats a reader of the number needs:
///
/// - It reports each buffer's ALLOCATION CAPACITY, not its logical length.
///   These are equal for both of this project's drivers (`Buffer::from_vec`
///   inherits the `Vec`'s exact layout and `MutableBuffer::from_len_zeroed`
///   allocates exactly `len`), but an over-allocated buffer reads high, and a
///   sliced array reports the whole allocation it is a window into.
/// - A buffer shared by two columns of the same batch is counted twice. No
///   batch in this workspace does that.
pub fn payload_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|c| c.get_buffer_memory_size())
        .sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int64Array, LargeBinaryArray, LargeListArray,
        StringArray, UInt32Array,
    };
    use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
    use arrow::datatypes::{DataType, Field, FieldRef, Schema};

    use super::*;
    use crate::clock::{HostTime, SensorTime, Tov};

    /// The dimensions and byte counts this project actually replays, so the
    /// assertions below are checkable against the dataset rather than against
    /// themselves. KITTI cam0 is 1242x375 RGB8; the sweep size is one of
    /// drive_0005's.
    const CAM_W: u32 = 1242;
    const CAM_H: u32 = 375;
    const CAM_PIXEL_BYTES: usize = 1_397_250;
    const SWEEP_POINTS: usize = 123_397;
    const POINT_BYTES: usize = 16;
    const SWEEP_BYTES: usize = 1_974_352;

    /// One LargeBinary column and nothing else: the only buffers are the value
    /// bytes and the offset pair, so the expected total is arithmetic.
    fn blob_batch(n: usize) -> RecordBatch {
        let values = Buffer::from_vec(vec![0u8; n]);
        let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n as i64]));
        let arr = LargeBinaryArray::try_new(offsets, values, None).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "blob",
            DataType::LargeBinary,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(arr) as ArrayRef]).unwrap()
    }

    /// The shape `pipes_kitti::frame::build_cam0_batch` produces: the pixels
    /// plus four small metadata columns. Rebuilt here because `pipes-core`
    /// cannot depend on `pipes-kitti`.
    fn cam0_like(w: u32, h: u32) -> RecordBatch {
        let n = w as usize * h as usize * 3;
        let values = Buffer::from_vec(vec![0u8; n]);
        let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n as i64]));
        let pixels = LargeBinaryArray::try_new(offsets, values, None).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("width", DataType::UInt32, false),
            Field::new("height", DataType::UInt32, false),
            Field::new("stride", DataType::UInt32, false),
            Field::new("pixel_format", DataType::Utf8, false),
            Field::new("pixels", DataType::LargeBinary, false),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(vec![w])),
            Arc::new(UInt32Array::from(vec![h])),
            Arc::new(UInt32Array::from(vec![w * 3])),
            Arc::new(StringArray::from(vec!["rgb8"])),
            Arc::new(pixels),
        ];
        RecordBatch::try_new(schema, columns).unwrap()
    }

    /// The shape `pipes_kitti::velo::build_velo_batch` produces: the points are
    /// two list levels down, which is the case a top-level-only size would get
    /// wrong by five orders of magnitude.
    fn velo_like(points: usize) -> RecordBatch {
        let value_field: FieldRef = Arc::new(Field::new("item", DataType::Float32, false));
        let point_field: FieldRef = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(Arc::clone(&value_field), 4),
            false,
        ));
        // `MutableBuffer`, as the driver does: `Float32Array` needs 4-byte
        // alignment and `Vec<u8>` does not promise it.
        let buf = Buffer::from(MutableBuffer::from_len_zeroed(points * POINT_BYTES));
        let values = Float32Array::new(ScalarBuffer::<f32>::from(buf), None);
        let point = FixedSizeListArray::try_new(value_field, 4, Arc::new(values) as ArrayRef, None)
            .unwrap();
        let offsets =
            OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, points as i64]));
        let points_arr =
            LargeListArray::try_new(Arc::clone(&point_field), offsets, Arc::new(point), None)
                .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("point_count", DataType::UInt32, false),
            Field::new("point_format", DataType::Utf8, false),
            Field::new("tov_trigger_ns", DataType::Int64, false),
            Field::new(
                "points",
                DataType::LargeList(Arc::clone(&point_field)),
                false,
            ),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(vec![points as u32])),
            Arc::new(StringArray::from(vec!["xyzr_f32le"])),
            Arc::new(Int64Array::from(vec![1_234i64])),
            Arc::new(points_arr),
        ];
        RecordBatch::try_new(schema, columns).unwrap()
    }

    fn sample_of(payload: RecordBatch) -> Sample {
        Sample {
            stream: StreamId::CAM0,
            seq: 0,
            arrival_seq: 0,
            parent: None,
            tov: Tov::Time(SensorTime(0)),
            epoch: 0,
            due: None,
            arrival: HostTime(0),
            payload,
            storage_id: 0,
            decode_ns: 0,
        }
    }

    #[test]
    fn names() {
        assert_eq!(StreamId::CAM0.name(), "cam0");
        assert_eq!(StreamId::TRACKS.name(), "tracks");
        assert_eq!(StreamId(99).name(), "unknown");
    }

    #[test]
    fn payload_bytes_is_the_buffers_and_only_the_buffers() {
        // 8 x 4 RGB8 = 96 B of payload. The batch's only other buffer is the
        // LargeBinary offset pair, two i64 = 16 B. 96 + 16 = 112, by hand.
        let batch = blob_batch(8 * 4 * 3);
        assert_eq!(payload_bytes(&batch), 112);
        assert_eq!(sample_of(batch.clone()).payload_bytes(), 112);
        // Strictly more than the buffers once arrow adds the Rust struct they
        // hang off. If `payload_bytes` were `get_array_memory_size` the number
        // would drift with the column count instead of with the data.
        assert!(
            batch.get_array_memory_size() > 112,
            "get_array_memory_size is supposed to include more than the buffers"
        );
    }

    #[test]
    fn an_empty_payload_carries_almost_nothing() {
        // Negative control for the test above: same schema, no data, so the
        // number must collapse. A size that ignored its argument would not.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "blob",
            DataType::LargeBinary,
            false,
        )]));
        let n = payload_bytes(&RecordBatch::new_empty(schema));
        assert!(n < 112, "an empty batch reported {n} B");
    }

    #[test]
    fn payload_bytes_tracks_the_payload_byte_for_byte() {
        let small = payload_bytes(&blob_batch(1_000));
        let big = payload_bytes(&blob_batch(1_000_000));
        assert_eq!(big - small, 999_000, "the difference is the payload");
    }

    #[test]
    fn a_kitti_camera_frame_reports_its_pixel_buffer() {
        assert_eq!(CAM_W as usize * CAM_H as usize * 3, CAM_PIXEL_BYTES);
        let full = payload_bytes(&cam0_like(CAM_W, CAM_H));
        assert!(full >= CAM_PIXEL_BYTES, "{full} < {CAM_PIXEL_BYTES}");
        let overhead = full - CAM_PIXEL_BYTES;
        println!(
            "cam0 {CAM_W}x{CAM_H}: payload_bytes={full} pixels={CAM_PIXEL_BYTES} overhead={overhead}"
        );
        // The excess is the four metadata columns and the offset pair, and it
        // is the SAME for a tiny frame: fixed, not a fraction of the payload.
        let tiny = payload_bytes(&cam0_like(64, 8));
        assert_eq!(
            tiny - 64 * 8 * 3,
            overhead,
            "overhead varies with resolution"
        );
        assert!(
            overhead < 512,
            "overhead {overhead} B is not per-batch bookkeeping"
        );
    }

    #[test]
    fn a_kitti_lidar_sweep_reports_its_point_buffer_through_the_nesting() {
        assert_eq!(SWEEP_POINTS * POINT_BYTES, SWEEP_BYTES);
        let full = payload_bytes(&velo_like(SWEEP_POINTS));
        assert!(
            full >= SWEEP_BYTES,
            "{full} < {SWEEP_BYTES}: the points are two list levels down"
        );
        let overhead = full - SWEEP_BYTES;
        println!(
            "lidar {SWEEP_POINTS} points: payload_bytes={full} points={SWEEP_BYTES} overhead={overhead}"
        );
        let tiny = payload_bytes(&velo_like(4));
        assert_eq!(
            tiny - 4 * POINT_BYTES,
            overhead,
            "overhead varies with sweep size"
        );
        assert!(
            overhead < 512,
            "overhead {overhead} B is not per-batch bookkeeping"
        );
    }

    #[test]
    fn a_shared_payload_is_that_size_on_every_edge() {
        // What the dashboard draws: one buffer, two consumers, and each edge
        // carries the whole sweep. The size is a property of the transfer, not
        // of who allocated it - the clone copies no points.
        let a = sample_of(velo_like(SWEEP_POINTS));
        let b = a.clone();
        assert_eq!(a.payload_bytes(), b.payload_bytes());
        assert!(a.payload_bytes() >= SWEEP_BYTES);
    }
}
