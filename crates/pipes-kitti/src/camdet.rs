//! The frozen camera detector: YOLOX-Nano run on each camera frame, and the
//! `CAM_DET` batch its detections travel in.
//!
//! **Frozen** means three things, each checked rather than asserted: the
//! weights are one file whose sha256 is pinned here ([`MODEL_SHA256`]) and
//! compared at load; nothing in this module learns or adapts; and the runtime
//! (`tract-onnx`, pinned exactly) runs single-threaded, so the same frame
//! gives the same bytes on every run. Which frames it is given is not this
//! module's to promise: on a loaded host the queue in front of it can evict
//! one.
//!
//! Every number below is the model's, not a tuning of this project's: the
//! input shape, the letterbox fill, the channel order, the score threshold
//! and the NMS overlap are the ones the model was published and evaluated
//! with, adapted only in the input shape — 640x192 keeps the KITTI frame's
//! aspect ratio instead of padding 70 % of a 416x416 square. They were
//! checked against the model's reference implementation (ONNX Runtime and
//! YOLOX's own preprocessing) on drive_0005 when the detector was chosen, and
//! the facts that matter are restated at each constant rather than left in
//! that investigation.
//!
//! The pipeline is: read the frame's shared pixel buffer **in place**
//! ([`preprocess`] borrows it), letterbox it into a new tensor, run the
//! network and [`decode`] every anchor above the threshold straight out of
//! its output ([`Model::infer`]), [`nms`] the survivors, and build one batch
//! per frame ([`build_cam_det_batch`]). The resized image and the tensor are
//! real copies — the network cannot read an RGB8 buffer — and the stage's
//! `bytes_alloc` reports them rather than hiding them.

use std::cmp::Ordering;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Float32Array, Int64Array, LargeListArray, RecordBatch, StringArray,
    StructArray, UInt32Array,
};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Fields, Float32Type, Schema, UInt32Type};
use image::{ImageBuffer, Rgb};
use sha2::{Digest, Sha256};
use tract_onnx::prelude::{
    tract_ndarray, tvec, DatumExt, Framework, InferenceModelExt, IntoRunnable, Tensor,
    TypedRunnableModel,
};

use crate::track::CamRef;

/// Where `pipes run` looks for the weights, relative to the working directory
/// (the repository root, where `runs/` also goes).
pub const MODEL_FILE: &str = "models/yolox_nano.onnx";

/// The script that puts [`MODEL_FILE`] there. Named by every error that
/// means "the model is not usable", so the message ends at a next step.
pub const FETCH_SCRIPT: &str = "scripts/fetch_model.ps1";

/// sha256 of the one file this module will load: `yolox_nano.onnx` from
/// YOLOX release 0.1.1rc0, 3,659,407 bytes.
pub const MODEL_SHA256: &str = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d";

/// Model name, as `run.json` and the schema metadata record it.
pub const MODEL_NAME: &str = "yolox_nano";

/// Network input width in pixels.
///
/// 640x192 is the KITTI frame's aspect ratio (1242/375 = 3.31) rounded to the
/// network's stride of 32: r = 0.512, content 635x192, five columns of padding.
/// The published 416x416 would spend 70 % of its compute on padding and see
/// objects at the scale of a 416x128 input.
pub const INPUT_W: usize = 640;
/// Network input height in pixels. See [`INPUT_W`].
pub const INPUT_H: usize = 192;

/// Value of every letterbox pixel outside the resized image, per channel.
/// YOLOX's own preprocessing fills with 114.
pub const PAD_VALUE: f32 = 114.0;

/// A detection is kept when objectness x class probability is ABOVE this.
/// YOLOX's demo threshold for its published results is 0.3.
pub const SCORE_THRESHOLD: f32 = 0.3;

/// Greedy NMS suppresses a box whose IoU with a higher-scored kept box is
/// ABOVE this. YOLOX's reference value, applied class-agnostically as its
/// reference is — which is what removes the "truck" drawn over a "car".
pub const NMS_IOU: f32 = 0.45;

/// Columns per anchor in the network output: 4 box terms, objectness, 80
/// class probabilities.
pub const OUTPUT_COLS: usize = 85;

/// The three strides of YOLOX's heads, in output-row order.
pub const STRIDES: [usize; 3] = [8, 16, 32];

/// Anchors (output rows) at [`INPUT_W`]x[`INPUT_H`]: 1920 + 480 + 120.
pub const ANCHORS: usize = (INPUT_W / 8) * (INPUT_H / 8)
    + (INPUT_W / 16) * (INPUT_H / 16)
    + (INPUT_W / 32) * (INPUT_H / 32);

/// Value of the `ref_format` column of a detection batch.
///
/// The column is the one [`crate::track::CAM_REF_FORMAT`] uses, so the four
/// frame-reference columns read the same way off either payload and the
/// fusion's pairing does not care which of the two it was handed.
pub const CAM_DET_FORMAT: &str = "cam_det_yolox_nano_v1";

/// The class names, indexed by the network's class id: YOLOX's
/// `COCO_CLASSES`, which is a 0-based list of 80 and NOT the original COCO
/// category ids (which skip numbers).
pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// The name of a class id, or `"?"` for one the model cannot produce.
pub fn class_name(id: u32) -> &'static str {
    COCO_CLASSES.get(id as usize).copied().unwrap_or("?")
}

/// One detection, in the camera frame's own pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Det {
    /// Left edge, pixels, clipped to the image.
    pub x0: f32,
    /// Top edge.
    pub y0: f32,
    /// Right edge.
    pub x1: f32,
    /// Bottom edge.
    pub y1: f32,
    /// Index into [`COCO_CLASSES`].
    pub class_id: u32,
    /// Objectness x class probability, in (0.3, 1].
    pub score: f32,
    /// The output row it came from: the tie-break that makes the order total.
    pub row: u32,
}

impl Det {
    fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
}

/// Everything that can stop the detector from loading or running.
#[derive(Debug)]
pub enum CamDetError {
    /// The weights are not where they are looked for.
    Missing(PathBuf),
    /// The file exists and could not be read.
    Io(PathBuf, std::io::Error),
    /// The file is not the pinned model.
    Sha256Mismatch {
        /// The file that was read.
        path: PathBuf,
        /// What its bytes hash to.
        got: String,
    },
    /// The runtime refused the model or the input.
    Runtime(String),
    /// The frame is not `width * height * 3` bytes, or is empty.
    BadFrame {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
        /// Bytes supplied.
        got: usize,
    },
}

impl std::fmt::Display for CamDetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fetch = format!(
            "run `powershell -ExecutionPolicy Bypass -File {}` from the repository root \
             to download it, or pass `--detector off`",
            FETCH_SCRIPT.replace('/', "\\")
        );
        match self {
            CamDetError::Missing(p) => write!(
                f,
                "the camera detector's model is missing: {} does not exist; {fetch}",
                p.display()
            ),
            CamDetError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            CamDetError::Sha256Mismatch { path, got } => write!(
                f,
                "{} is not the pinned camera detector: sha256 {got}, expected {MODEL_SHA256}; \
                 delete it and {fetch}",
                path.display()
            ),
            CamDetError::Runtime(e) => write!(f, "camera detector: {e}"),
            CamDetError::BadFrame { width, height, got } => write!(
                f,
                "camera frame {width}x{height} has {got} bytes, not width * height * 3"
            ),
        }
    }
}

impl std::error::Error for CamDetError {}

/// Lower-case hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The loaded network. Built once, before the clock starts; `run` is the only
/// thing a frame costs.
pub struct Model {
    plan: Arc<TypedRunnableModel>,
    /// sha256 of the file it was loaded from — equal to [`MODEL_SHA256`], or
    /// [`Model::load`] would have refused it.
    pub sha256: String,
}

impl Model {
    /// Reads `path`, checks its sha256 against [`MODEL_SHA256`], and builds the
    /// runnable plan at 1x3x[`INPUT_H`]x[`INPUT_W`].
    ///
    /// The file declares a 416x416 input; `with_ignore_value_info` and
    /// `with_ignore_output_shapes` drop those declared shapes so the input
    /// fact can be replaced (its reshapes are `[1, 85, -1]`, so any multiple
    /// of 32 works). Without them tract fails to unify 640 with 416.
    pub fn load(path: &Path) -> Result<Model, CamDetError> {
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(CamDetError::Missing(path.to_path_buf()))
            }
            Err(e) => return Err(CamDetError::Io(path.to_path_buf(), e)),
        };
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| CamDetError::Io(path.to_path_buf(), e))?;
        let sha256 = sha256_hex(&bytes);
        if sha256 != MODEL_SHA256 {
            return Err(CamDetError::Sha256Mismatch {
                path: path.to_path_buf(),
                got: sha256,
            });
        }
        let rt = |e: tract_onnx::prelude::TractError| CamDetError::Runtime(format!("{e:#}"));
        let plan = tract_onnx::onnx()
            .with_ignore_value_info(true)
            .with_ignore_output_shapes(true)
            .model_for_read(&mut bytes.as_slice())
            .map_err(rt)?
            .with_input_fact(0, f32::fact([1, 3, INPUT_H, INPUT_W]).into())
            .map_err(rt)?
            .into_optimized()
            .map_err(rt)?
            .into_runnable()
            .map_err(rt)?;
        Ok(Model { plan, sha256 })
    }

    /// Runs the network on one tensor from [`preprocess`], which it consumes,
    /// and [`decode`]s the `[ANCHORS x 85]` output **where the runtime left
    /// it**: the 857 KB output is read in place, never copied out.
    ///
    /// `r`, `width` and `height` are the letterbox ratio [`preprocess`]
    /// returned and the frame's own size, which the boxes are mapped back to.
    /// The candidates come back in output-row order, before [`nms`].
    pub fn infer(
        &self,
        input: Vec<f32>,
        r: f64,
        width: u32,
        height: u32,
    ) -> Result<Vec<Det>, CamDetError> {
        let rt = |e: tract_onnx::prelude::TractError| CamDetError::Runtime(format!("{e:#}"));
        // Moved, not copied: the tensor takes the Vec's buffer.
        let t: Tensor = tract_ndarray::Array4::from_shape_vec((1, 3, INPUT_H, INPUT_W), input)
            .map_err(|e| CamDetError::Runtime(e.to_string()))?
            .into();
        let out = self.plan.run(tvec!(t.into())).map_err(rt)?;
        let first = out
            .first()
            .ok_or_else(|| CamDetError::Runtime("no output".to_string()))?;
        let view = first.to_plain_array_view::<f32>().map_err(rt)?;
        // A tract tensor owns one contiguous, standard-layout buffer, so the
        // view is a slice and the fallback copy is never expected to run; it
        // exists so a layout change would cost time rather than correctness.
        let copied: Vec<f32>;
        let raw: &[f32] = match view.as_slice() {
            Some(s) => s,
            None => {
                copied = view.iter().copied().collect();
                &copied
            }
        };
        if raw.len() != ANCHORS * OUTPUT_COLS {
            return Err(CamDetError::Runtime(format!(
                "output has {} values, expected {ANCHORS} x {OUTPUT_COLS}",
                raw.len()
            )));
        }
        Ok(decode(raw, r, width, height))
    }
}

/// The letterbox scale for a `width x height` frame: the largest `r` that fits
/// the frame inside the input, as YOLOX computes it (f64, then truncated).
pub fn letterbox_ratio(width: u32, height: u32) -> f64 {
    (INPUT_H as f64 / f64::from(height)).min(INPUT_W as f64 / f64::from(width))
}

/// The resized content size inside the letterbox: `trunc(w r) x trunc(h r)`.
pub fn letterbox_size(width: u32, height: u32) -> (u32, u32) {
    let r = letterbox_ratio(width, height);
    (
        ((f64::from(width) * r) as u32).clamp(1, INPUT_W as u32),
        ((f64::from(height) * r) as u32).clamp(1, INPUT_H as u32),
    )
}

/// The network input for one RGB8 frame, read **in place** from `rgb`.
///
/// Resize bilinear (`Triangle`, which reproduces the reference cv2
/// `INTER_LINEAR` detections to within 0.01 of score) to
/// [`letterbox_size`], place it at the TOP-LEFT of a [`INPUT_W`]x[`INPUT_H`]
/// canvas filled with [`PAD_VALUE`], planes in B, G, R order, raw 0..255
/// values, NCHW. Returns the tensor and the ratio [`decode`] divides by.
///
/// Three allocations, all real and all reported by the stage: the resize's
/// own intermediate (its vertical pass writes a four-channel f32 image of the
/// frame's width by the new height, 1242 x 192 x 16 = 3,815,424 bytes on
/// KITTI), the resized image (`trunc(w r) * trunc(h r) * 3` = 365,760
/// bytes) and the tensor (`3 * 192 * 640 * 4` = 1,474,560 bytes) -- 5,655,744
/// bytes. A run on drive_0005 measures 5,655,808 per frame: 64 bytes more,
/// not traced.
pub fn preprocess(rgb: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, f64), CamDetError> {
    let bad = CamDetError::BadFrame {
        width,
        height,
        got: rgb.len(),
    };
    if width == 0 || height == 0 || rgb.len() != width as usize * height as usize * 3 {
        return Err(bad);
    }
    let src: ImageBuffer<Rgb<u8>, &[u8]> = ImageBuffer::from_raw(width, height, rgb).ok_or(bad)?;
    let r = letterbox_ratio(width, height);
    let (nw, nh) = letterbox_size(width, height);
    let resized = image::imageops::resize(&src, nw, nh, image::imageops::FilterType::Triangle);
    let plane = INPUT_W * INPUT_H;
    let mut t = vec![PAD_VALUE; 3 * plane];
    let (nw, nh) = (nw as usize, nh as usize);
    let raw = resized.as_raw();
    for y in 0..nh {
        let row = &raw[y * nw * 3..(y + 1) * nw * 3];
        let (px, _) = row.as_chunks::<3>();
        let base = y * INPUT_W;
        for (x, p) in px.iter().enumerate() {
            t[base + x] = f32::from(p[2]);
            t[plane + base + x] = f32::from(p[1]);
            t[2 * plane + base + x] = f32::from(p[0]);
        }
    }
    Ok((t, r))
}

/// Every anchor whose score is above [`SCORE_THRESHOLD`], as a box in the
/// frame's own pixels.
///
/// The output is `[ANCHORS x 85]`, stride-8 cells first (row-major), then
/// stride 16, then 32. Columns 0..3 are grid-relative `(dx, dy, log w, log
/// h)`; 4 is objectness and 5..84 the class probabilities, both already
/// through a sigmoid. `cx = (dx + gx) s`, `w = exp(log w) s`; corners divided
/// by `r` and clipped to the image. Class = argmax (the first on a tie).
pub fn decode(raw: &[f32], r: f64, width: u32, height: u32) -> Vec<Det> {
    let mut out = Vec::new();
    let (wf, hf) = (width as f32, height as f32);
    let r = r as f32;
    let mut row = 0usize;
    for s in STRIDES {
        let (gw, gh) = (INPUT_W / s, INPUT_H / s);
        for gy in 0..gh {
            for gx in 0..gw {
                let i = row;
                row += 1;
                let Some(a) = raw.get(i * OUTPUT_COLS..(i + 1) * OUTPUT_COLS) else {
                    return out;
                };
                let obj = a[4];
                let mut best = 0usize;
                for (k, v) in a[5..].iter().enumerate() {
                    if *v > a[5 + best] {
                        best = k;
                    }
                }
                let score = obj * a[5 + best];
                // Positive test: a NaN score is never kept.
                if score.partial_cmp(&SCORE_THRESHOLD) != Some(Ordering::Greater) {
                    continue;
                }
                let sf = s as f32;
                let cx = (a[0] + gx as f32) * sf;
                let cy = (a[1] + gy as f32) * sf;
                let w = a[2].exp() * sf;
                let h = a[3].exp() * sf;
                out.push(Det {
                    x0: ((cx - w / 2.0) / r).clamp(0.0, wf),
                    y0: ((cy - h / 2.0) / r).clamp(0.0, hf),
                    x1: ((cx + w / 2.0) / r).clamp(0.0, wf),
                    y1: ((cy + h / 2.0) / r).clamp(0.0, hf),
                    class_id: best as u32,
                    score,
                    row: i as u32,
                });
            }
        }
    }
    out
}

/// Intersection over union of two boxes; 0 when either is empty.
pub fn iou(a: &Det, b: &Det) -> f32 {
    let iw = (a.x1.min(b.x1) - a.x0.max(b.x0)).max(0.0);
    let ih = (a.y1.min(b.y1) - a.y0.max(b.y0)).max(0.0);
    let inter = iw * ih;
    let union = a.area() + b.area() - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Class-agnostic greedy NMS, and the order the batch is written in.
///
/// Sorted by score descending, then by output row ascending — a total order,
/// so equal scores cannot swap between runs — and a box is kept unless its
/// IoU with an already-kept box is above [`NMS_IOU`]. The result is in that
/// same order, which is what makes the `CAM_DET` payload byte-stable.
pub fn nms(mut dets: Vec<Det>) -> Vec<Det> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.row.cmp(&b.row)));
    let mut kept: Vec<Det> = Vec::with_capacity(dets.len());
    for d in dets {
        if kept
            .iter()
            .all(|k| iou(k, &d).partial_cmp(&NMS_IOU) != Some(Ordering::Greater))
        {
            kept.push(d);
        }
    }
    kept
}

/// Everything from pixels to the final detections, for one frame.
pub fn detect_frame(
    model: &Model,
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<Det>, CamDetError> {
    let (t, r) = preprocess(rgb, width, height)?;
    Ok(nms(model.infer(t, r, width, height)?))
}

fn det_fields() -> Fields {
    Fields::from(vec![
        Field::new("x0", DataType::Float32, false),
        Field::new("y0", DataType::Float32, false),
        Field::new("x1", DataType::Float32, false),
        Field::new("y1", DataType::Float32, false),
        Field::new("class_id", DataType::UInt32, false),
        Field::new("class_name", DataType::Utf8, false),
        Field::new("confidence", DataType::Float32, false),
    ])
}

fn det_item_field() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Struct(det_fields()), false))
}

/// `frame_seq i64, tov_ns i64, width u32, height u32, ref_format Utf8,
/// detections LargeList<Struct<x0 f32, y0 f32, x1 f32, y1 f32, class_id u32,
/// class_name Utf8, confidence f32>>`, one row per camera frame.
///
/// **The first four columns are the frame reference** the fusion pairs on,
/// with the same names and meaning as [`crate::track::cam_ref_schema`]'s, so
/// the detector's output replaces that payload on the same stream rather than
/// travelling beside it. A frame with no detections is still a row, with an
/// empty list: "the camera looked and saw nothing" is a result, and the
/// fusion needs the frame's instant either way.
///
/// **Provenance** of the model is in the schema's metadata — `model`,
/// `model_sha256`, `input`, `score_threshold`, `nms_iou` — which every batch
/// carries by reference at no per-frame cost. The frame's own seq and instant
/// are the first two columns; the sample's `parent` names it too.
pub fn cam_det_schema(model_sha256: &str) -> Arc<Schema> {
    let meta = [
        ("model", MODEL_NAME.to_string()),
        ("model_sha256", model_sha256.to_string()),
        (
            "input",
            format!("1x3x{INPUT_H}x{INPUT_W} bgr letterbox-top-left"),
        ),
        ("score_threshold", SCORE_THRESHOLD.to_string()),
        ("nms_iou", NMS_IOU.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    Arc::new(
        Schema::new(vec![
            Field::new("frame_seq", DataType::Int64, false),
            Field::new("tov_ns", DataType::Int64, false),
            Field::new("width", DataType::UInt32, false),
            Field::new("height", DataType::UInt32, false),
            Field::new("ref_format", DataType::Utf8, false),
            Field::new("detections", DataType::LargeList(det_item_field()), false),
        ])
        .with_metadata(meta),
    )
}

/// Builds the one-row `CAM_DET` batch for a frame and its detections.
pub fn build_cam_det_batch(
    r: CamRef,
    dets: &[Det],
    schema: &Arc<Schema>,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let f = |g: fn(&Det) -> f32| Arc::new(Float32Array::from_iter_values(dets.iter().map(g)));
    let columns: Vec<ArrayRef> = vec![
        f(|d| d.x0),
        f(|d| d.y0),
        f(|d| d.x1),
        f(|d| d.y1),
        Arc::new(UInt32Array::from_iter_values(
            dets.iter().map(|d| d.class_id),
        )),
        Arc::new(StringArray::from_iter_values(
            dets.iter().map(|d| class_name(d.class_id)),
        )),
        f(|d| d.score),
    ];
    let items = StructArray::try_new(det_fields(), columns, None)?;
    let offsets =
        OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, dets.len() as i64]));
    let list = LargeListArray::try_new(det_item_field(), offsets, Arc::new(items), None)?;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![r.seq as i64])),
        Arc::new(Int64Array::from(vec![r.tov_ns])),
        Arc::new(UInt32Array::from(vec![r.width])),
        Arc::new(UInt32Array::from(vec![r.height])),
        Arc::new(StringArray::from(vec![CAM_DET_FORMAT])),
        Arc::new(list),
    ];
    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// The address of a `CAM_DET` batch's detection offsets: the buffer the
/// fusion reads the batch through, and so the `storage_id` its producer
/// records and its consumer re-derives. `None` for anything that is not a
/// detection batch, including a bare frame reference, which names no buffer.
///
/// The offsets rather than a detection column, because they are a real
/// allocation on every batch: a frame with no detections has an empty `x0`
/// column, whose address is a dangling placeholder shared by every empty
/// array, and two batches would then "share" a buffer neither has.
pub fn cam_det_storage_id(batch: &RecordBatch) -> Option<usize> {
    let fmt = batch.column_by_name("ref_format")?.as_string_opt::<i32>()?;
    if fmt.is_empty() || fmt.value(0) != CAM_DET_FORMAT {
        return None;
    }
    let list = batch.column_by_name("detections")?.as_list_opt::<i64>()?;
    Some(list.offsets().inner().inner().as_ptr() as usize)
}

/// The detections of one `CAM_DET` batch, **borrowed where the batch holds
/// them**: one slice per column, all the same length, index `i` of each being
/// detection `i` of the frame. Nothing is copied and nothing is allocated, so
/// the fusion can read a frame's detections inside its measured window and
/// still report 0 for reading them.
///
/// The index is the detection's rank in the batch -- score descending, then
/// the network's output row ([`nms`]) -- which is a total order, so "detection
/// 3 of frame 41" names the same box on every run and in `cam_det.arrows`.
#[derive(Clone, Copy, Debug)]
pub struct DetView<'a> {
    /// Left edges, pixels.
    pub x0: &'a [f32],
    /// Top edges.
    pub y0: &'a [f32],
    /// Right edges.
    pub x1: &'a [f32],
    /// Bottom edges.
    pub y1: &'a [f32],
    /// Indices into [`COCO_CLASSES`].
    pub class_id: &'a [u32],
    /// Scores, in (0.3, 1].
    pub confidence: &'a [f32],
}

impl DetView<'_> {
    /// Detections in the frame.
    pub fn len(&self) -> usize {
        self.x0.len()
    }

    /// Whether the detector found nothing in the frame.
    pub fn is_empty(&self) -> bool {
        self.x0.is_empty()
    }

    /// Detection `i` as a [`Det`] whose `row` is `i`, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<Det> {
        Some(Det {
            x0: *self.x0.get(i)?,
            y0: *self.y0.get(i)?,
            x1: *self.x1.get(i)?,
            y1: *self.y1.get(i)?,
            class_id: *self.class_id.get(i)?,
            score: *self.confidence.get(i)?,
            row: u32::try_from(i).ok()?,
        })
    }
}

/// The detections of a `CAM_DET` batch as a [`DetView`], or `None` if the
/// batch is not a detection batch -- including a bare frame reference, which
/// carries none -- or is not the one row per frame this module builds.
///
/// Allocates nothing: the struct's children are found by position among its
/// fields rather than through `StructArray::column_by_name`, which builds a
/// `Vec` of the names on every call.
pub fn cam_det_view(batch: &RecordBatch) -> Option<DetView<'_>> {
    let fmt = batch.column_by_name("ref_format")?.as_string_opt::<i32>()?;
    if fmt.is_empty() || fmt.value(0) != CAM_DET_FORMAT {
        return None;
    }
    let list = batch.column_by_name("detections")?.as_list_opt::<i64>()?;
    if list.len() != 1 {
        return None;
    }
    let offsets = list.value_offsets();
    let (a, b) = (
        usize::try_from(*offsets.first()?).ok()?,
        usize::try_from(*offsets.get(1)?).ok()?,
    );
    let s = list.values().as_struct_opt()?;
    let child = |name: &str| {
        s.fields()
            .iter()
            .position(|f| f.name() == name)
            .map(|i| s.column(i))
    };
    let f32s = |name: &str| -> Option<&[f32]> {
        let values: &[f32] = child(name)?.as_primitive_opt::<Float32Type>()?.values();
        values.get(a..b)
    };
    let class_id: &[u32] = child("class_id")?
        .as_primitive_opt::<UInt32Type>()?
        .values();
    Some(DetView {
        x0: f32s("x0")?,
        y0: f32s("y0")?,
        x1: f32s("x1")?,
        y1: f32s("y1")?,
        class_id: class_id.get(a..b)?,
        confidence: f32s("confidence")?,
    })
}

/// The detections of a `CAM_DET` batch, or `None` if the batch is not one.
///
/// `row` is the detection's index in the batch (its rank), since the output
/// row it came from is not carried.
pub fn read_cam_dets(batch: &RecordBatch) -> Option<Vec<Det>> {
    let fmt = batch.column_by_name("ref_format")?.as_string_opt::<i32>()?;
    if fmt.is_empty() || fmt.value(0) != CAM_DET_FORMAT {
        return None;
    }
    let list = batch
        .column_by_name("detections")?
        .as_list_opt::<i64>()?
        .value(0);
    let s = list.as_struct_opt()?;
    let col = |name: &str| s.column_by_name(name)?.as_primitive_opt::<Float32Type>();
    let (x0, y0, x1, y1, conf) = (
        col("x0")?,
        col("y0")?,
        col("x1")?,
        col("y1")?,
        col("confidence")?,
    );
    let cls = s
        .column_by_name("class_id")?
        .as_primitive_opt::<UInt32Type>()?;
    Some(
        (0..s.len())
            .map(|i| Det {
                x0: x0.value(i),
                y0: y0.value(i),
                x1: x1.value(i),
                y1: y1.value(i),
                class_id: cls.value(i),
                score: conf.value(i),
                row: i as u32,
            })
            .collect(),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// One output row with the given box terms, objectness and one class.
    fn anchor(raw: &mut [f32], i: usize, b: [f32; 4], obj: f32, class: usize, p: f32) {
        let a = &mut raw[i * OUTPUT_COLS..(i + 1) * OUTPUT_COLS];
        a[..4].copy_from_slice(&b);
        a[4] = obj;
        a[5 + class] = p;
    }

    fn det(x0: f32, y0: f32, x1: f32, y1: f32, score: f32, row: u32) -> Det {
        Det {
            x0,
            y0,
            x1,
            y1,
            class_id: 2,
            score,
            row,
        }
    }

    #[test]
    fn the_kitti_frame_letterboxes_to_635_by_192() {
        assert_eq!(ANCHORS, 2520);
        assert_eq!(letterbox_size(1242, 375), (635, 192));
        assert!((letterbox_ratio(1242, 375) - 0.512).abs() < 1e-12);
    }

    #[test]
    fn preprocessing_puts_bgr_planes_top_left_and_pads_with_114() {
        // A frame already at the network's size, so the resize is the
        // identity and every tensor value is predictable.
        let (w, h) = (INPUT_W as u32, INPUT_H as u32);
        let mut rgb = vec![0u8; w as usize * h as usize * 3];
        for (i, p) in rgb.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            p.copy_from_slice(&[10, 20, (i % 200) as u8]);
        }
        let (t, r) = preprocess(&rgb, w, h).unwrap();
        assert_eq!(r, 1.0);
        let plane = INPUT_W * INPUT_H;
        assert_eq!(t.len(), 3 * plane);
        // B first, then G, then R.
        assert_eq!(t[207], 7.0);
        assert_eq!(t[plane + 207], 20.0);
        assert_eq!(t[2 * plane + 207], 10.0);

        // Half the width: the right half is padding, the left half is image,
        // and nothing is centred.
        let (w2, h2) = (INPUT_W as u32 / 2, INPUT_H as u32 / 2);
        let rgb2 = vec![200u8; w2 as usize * h2 as usize * 3];
        let (t2, r2) = preprocess(&rgb2, w2, h2).unwrap();
        assert_eq!(r2, 2.0);
        // Upscaled to fill 640x192 exactly: no padding at all.
        assert!(t2.iter().all(|v| *v == 200.0));

        // A frame twice as wide as it is tall at the input's height: the
        // content is 384 wide, and every column right of it is 114.
        let rgb3 = vec![50u8; 384 * 192 * 3];
        let (t3, _) = preprocess(&rgb3, 384, 192).unwrap();
        for c in 0..3 {
            let p = &t3[c * plane..(c + 1) * plane];
            assert_eq!(p[0], 50.0, "top-left is image");
            assert_eq!(p[383], 50.0);
            assert_eq!(p[384], PAD_VALUE, "right of the content is padding");
            assert_eq!(p[plane - 1], PAD_VALUE);
        }
    }

    #[test]
    fn a_frame_of_the_wrong_length_is_refused() {
        assert!(matches!(
            preprocess(&[0u8; 10], 2, 2),
            Err(CamDetError::BadFrame { got: 10, .. })
        ));
        // Positive control: the right length is accepted.
        assert!(preprocess(&[0u8; 12], 2, 2).is_ok());
    }

    #[test]
    fn decode_places_a_box_from_its_grid_cell_and_stride() {
        let mut raw = vec![0f32; ANCHORS * OUTPUT_COLS];
        // Stride-8 cell (gx 3, gy 2) is row 2 * 80 + 3.
        let i = 2 * (INPUT_W / 8) + 3;
        anchor(&mut raw, i, [0.5, 0.5, 2f32.ln(), 0.0], 0.9, 2, 0.8);
        // Below threshold: 0.5 * 0.5 = 0.25.
        anchor(&mut raw, i + 1, [0.0, 0.0, 0.0, 0.0], 0.5, 0, 0.5);
        // The first stride-16 row is cell (0, 0) of that head.
        let j = (INPUT_W / 8) * (INPUT_H / 8);
        anchor(&mut raw, j, [0.0, 0.0, 0.0, 0.0], 1.0, 0, 0.4);
        let d = decode(&raw, 0.5, 1280, 384);
        assert_eq!(d.len(), 2, "{d:?}");
        // cx = 3.5 * 8 = 28, cy = 2.5 * 8 = 20, w = 16, h = 8; / 0.5.
        let a = d[0];
        assert_eq!((a.x0, a.y0, a.x1, a.y1), (40.0, 32.0, 72.0, 48.0));
        assert_eq!(a.class_id, 2);
        assert!((a.score - 0.72).abs() < 1e-6);
        assert_eq!(a.row, i as u32);
        // Clipped at the image's left and top edges: cx = 0, w = 16.
        let b = d[1];
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (0.0, 0.0, 16.0, 16.0));
        assert_eq!(b.row, j as u32);
    }

    #[test]
    fn a_score_exactly_at_the_threshold_or_nan_is_dropped() {
        let mut raw = vec![0f32; ANCHORS * OUTPUT_COLS];
        anchor(&mut raw, 0, [0.0; 4], 1.0, 0, SCORE_THRESHOLD);
        anchor(&mut raw, 1, [0.0; 4], f32::NAN, 0, 0.9);
        assert!(decode(&raw, 1.0, 640, 192).is_empty());
        // Positive control: just above it is kept.
        anchor(&mut raw, 0, [0.0; 4], 1.0, 0, SCORE_THRESHOLD + 1e-6);
        assert_eq!(decode(&raw, 1.0, 640, 192).len(), 1);
    }

    #[test]
    fn nms_removes_a_planted_duplicate_across_classes_and_keeps_the_rest() {
        let car = det(100.0, 100.0, 200.0, 200.0, 0.9, 5);
        let mut truck = det(105.0, 102.0, 205.0, 198.0, 0.5, 9);
        truck.class_id = 7;
        let other = det(400.0, 100.0, 450.0, 150.0, 0.6, 1);
        // IoU just under 0.45 with the car: kept (positive control).
        let neighbour = det(150.0, 100.0, 250.0, 200.0, 0.4, 2);
        assert!(iou(&car, &truck) > NMS_IOU);
        assert!(iou(&car, &neighbour) < NMS_IOU);
        let kept = nms(vec![truck, neighbour, other, car]);
        assert_eq!(kept, vec![car, other, neighbour]);
    }

    #[test]
    fn nms_order_is_total_on_equal_scores() {
        let a = det(0.0, 0.0, 10.0, 10.0, 0.5, 7);
        let b = det(100.0, 0.0, 110.0, 10.0, 0.5, 3);
        assert_eq!(nms(vec![a, b]), vec![b, a]);
        assert_eq!(nms(vec![b, a]), vec![b, a]);
    }

    #[test]
    fn the_batch_round_trips_and_carries_its_provenance() {
        let schema = cam_det_schema(MODEL_SHA256);
        assert_eq!(schema.metadata()["model_sha256"], MODEL_SHA256);
        let r = CamRef {
            seq: 12,
            tov_ns: 34,
            width: 1242,
            height: 375,
        };
        let mut dets = vec![
            det(1.0, 2.0, 3.0, 4.0, 0.9, 0),
            det(5.0, 6.0, 7.0, 8.0, 0.4, 1),
        ];
        dets[1].class_id = 0;
        let b = build_cam_det_batch(r, &dets, &schema).unwrap();
        assert_eq!(b.num_rows(), 1);
        assert_eq!(b.schema().metadata()["model"], MODEL_NAME);
        let back = read_cam_dets(&b).unwrap();
        assert_eq!(back, dets);
        let names = b
            .column_by_name("detections")
            .unwrap()
            .as_list::<i64>()
            .value(0);
        let names = names
            .as_struct()
            .column_by_name("class_name")
            .unwrap()
            .as_string::<i32>()
            .clone();
        assert_eq!(names.value(0), "car");
        assert_eq!(names.value(1), "person");
        // The frame reference reads the same way it does off a bare reference.
        assert_eq!(crate::track::read_cam_ref(&b), Some(r));

        // No detections is still a frame.
        let empty = build_cam_det_batch(r, &[], &schema).unwrap();
        assert_eq!(read_cam_dets(&empty), Some(vec![]));
        assert_eq!(crate::track::read_cam_ref(&empty), Some(r));

        // The storage id names the batch's own buffer: a clone (what a
        // consumer holds) reads the same address, and another batch -- even
        // an empty one -- does not share it.
        let id = cam_det_storage_id(&b).unwrap();
        assert_eq!(cam_det_storage_id(&b.clone()), Some(id));
        assert_ne!(cam_det_storage_id(&empty), Some(id));
        let empty2 = build_cam_det_batch(r, &[], &schema).unwrap();
        assert_ne!(cam_det_storage_id(&empty), cam_det_storage_id(&empty2));
    }

    /// The borrowed view reads what the copying reader reads, detection for
    /// detection, and refuses what it refuses.
    #[test]
    fn the_borrowed_view_is_the_batch_s_own_detections() {
        let schema = cam_det_schema(MODEL_SHA256);
        let r = CamRef {
            seq: 3,
            tov_ns: 4,
            width: 1242,
            height: 375,
        };
        let mut dets = vec![
            det(10.0, 20.0, 30.0, 40.0, 0.9, 0),
            det(50.0, 60.0, 70.0, 80.0, 0.5, 1),
            det(1.0, 2.0, 3.0, 4.0, 0.31, 2),
        ];
        dets[1].class_id = 0;
        let b = build_cam_det_batch(r, &dets, &schema).unwrap();
        let v = cam_det_view(&b).expect("a detection batch has a view");
        assert_eq!(v.len(), 3);
        let back: Vec<Det> = (0..v.len()).filter_map(|i| v.get(i)).collect();
        assert_eq!(back, read_cam_dets(&b).unwrap());
        assert_eq!(v.class_id, &[2, 0, 2]);
        assert!(v.get(3).is_none());
        // The slices are the batch's own buffer, not a copy of it.
        let x0 = b
            .column_by_name("detections")
            .unwrap()
            .as_list::<i64>()
            .values()
            .as_struct()
            .column(0)
            .as_primitive::<Float32Type>()
            .values()
            .as_ptr();
        assert_eq!(v.x0.as_ptr(), x0);
        // An empty frame is a view of nothing, not no view.
        let empty = build_cam_det_batch(r, &[], &schema).unwrap();
        assert!(cam_det_view(&empty).unwrap().is_empty());
        // A bare frame reference has no detections to view.
        let bare = crate::track::build_cam_ref_batch(r, &crate::track::cam_ref_schema()).unwrap();
        assert!(cam_det_view(&bare).is_none());
    }

    #[test]
    fn a_bare_frame_reference_is_not_read_as_detections() {
        let r = CamRef {
            seq: 1,
            tov_ns: 2,
            width: 3,
            height: 4,
        };
        let b = crate::track::build_cam_ref_batch(r, &crate::track::cam_ref_schema()).unwrap();
        assert_eq!(read_cam_dets(&b), None);
        assert_eq!(cam_det_storage_id(&b), None);
        // Positive control: the reference itself still reads.
        assert_eq!(crate::track::read_cam_ref(&b), Some(r));
    }

    #[test]
    fn a_missing_model_names_the_fetch_script() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("yolox_nano.onnx");
        let err = Model::load(&path).err().unwrap();
        assert!(matches!(err, CamDetError::Missing(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("fetch_model.ps1"), "{msg}");
        assert!(msg.contains("--detector off"), "{msg}");
        // A file that is there but is not the model is refused by its hash,
        // and the message names the script too.
        std::fs::write(&path, b"not a model").unwrap();
        let err = Model::load(&path).err().unwrap();
        assert!(matches!(err, CamDetError::Sha256Mismatch { .. }), "{err:?}");
        assert!(err.to_string().contains("fetch_model.ps1"));
    }

    #[test]
    fn sha256_is_the_standard_one() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
