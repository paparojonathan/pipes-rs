//! The frozen camera detector: a YOLOX network run on each camera frame, and
//! the `CAM_DET` batch its detections travel in.
//!
//! **Frozen** means three things, each checked rather than asserted: the
//! weights are one file whose sha256 is pinned here ([`ModelSpec::sha256`])
//! and compared at load; nothing in this module learns or adapts; and the
//! default runtime (`tract-onnx`, pinned exactly) runs single-threaded, so
//! the same frame gives the same bytes on every run. Which frames it is given
//! is not this module's to promise: on a loaded host the queue in front of it
//! can evict one.
//!
//! Six models and three runtimes ([`MODELS`], [`Backend`]). The models are
//! YOLOX-Nano, -S, -M, -L and -X as Megvii publishes them, each as a derived
//! file with its declared input set to the size the pipeline feeds
//! (`scripts/prepare_models.py`; ONNX Runtime refuses any other size, tract
//! ignores the declared one). The two biggest are for the GPU, at the
//! frame's own resolution and with their weights as 16-bit floats, which a
//! GPU runs half as fast again and which finds the same objects. The runtimes are tract on one thread, ONNX
//! Runtime on a CPU thread pool, and ONNX Runtime's CUDA provider on the
//! GPU; the second and third are loaded from the library `ORT_DYLIB_PATH`
//! names, so a machine without it still runs the default.
//!
//! Every number below is the model's, not a tuning of this project's: the
//! letterbox fill, the channel order, the score threshold and the NMS
//! overlap are the ones the model was published and evaluated with, adapted
//! only in the input shape — 640x192 keeps the KITTI frame's aspect ratio
//! instead of padding 70 % of a 416x416 square, and 1024x320 gives the
//! bigger models the frame at four fifths of its own resolution. They were
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
use std::sync::{Arc, Mutex, PoisonError};

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

use crate::onnx::ort_session;
use crate::track::CamRef;

/// Which runtime runs the network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// `tract`, pure Rust, one thread: the same frame gives the same bytes on
    /// every run, on any machine. The default.
    #[default]
    Tract,
    /// ONNX Runtime on the CPU, with an intra-op thread pool.
    Ort,
    /// ONNX Runtime's CUDA execution provider: the network on the GPU, with
    /// cuDNN's convolution algorithms chosen by its heuristics rather than
    /// by timing ([`CudaSearch::Heuristic`]) and TF32 off, so a run is
    /// repeatable on one machine.
    Cuda,
}

/// How the `cuda` backend picks each convolution's cuDNN algorithm.
///
/// A run uses [`CudaSearch::Heuristic`]: cuDNN's own ranking by shape, the
/// same every time on one machine, so two runs give the same bytes. The
/// others exist for `pipes bench-detector` to measure against it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CudaSearch {
    /// `cudnnGetConvolutionForwardAlgorithm_v7`: ranked by heuristics, no
    /// timing, deterministic.
    #[default]
    Heuristic,
    /// cuDNN's default algorithm for every convolution, whatever its shape:
    /// deterministic, and very slow for depthwise convolutions.
    Default,
    /// `cudnnFindConvolutionForwardAlgorithmEx`: every algorithm timed on
    /// the first frame and the fastest kept. Can pick differently from run
    /// to run.
    Exhaustive,
}

impl CudaSearch {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            CudaSearch::Heuristic => "heuristic",
            CudaSearch::Default => "default",
            CudaSearch::Exhaustive => "exhaustive",
        }
    }
}

/// The `cuda` backend's two knobs, both recorded by a run: how each
/// convolution's algorithm is chosen, and whether the GPU's tensor cores
/// may round the matmuls' inputs to TF32 (10 bits of mantissa). Off, the
/// arithmetic is single precision like the CPU's and the detections are the
/// CPU's to the last places the algorithms allow; on, faster on an Ampere
/// GPU and still the same bytes on every run of one machine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaOptions {
    /// The algorithm choice.
    pub search: CudaSearch,
    /// TF32 on the tensor cores.
    pub tf32: bool,
}

impl Backend {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Tract => "tract",
            Backend::Ort => "ort",
            Backend::Cuda => "cuda",
        }
    }

    /// Whether the backend is ONNX Runtime, loaded from [`ORT_DYLIB_ENV`].
    pub fn is_ort(self) -> bool {
        matches!(self, Backend::Ort | Backend::Cuda)
    }
}

/// The environment variable naming the ONNX Runtime library the `ort` and
/// `cuda` backends load (`libonnxruntime.so`, version 1.22.0, the GPU build
/// for `cuda`); [`ORT_DYLIB_DEFAULT`] when unset.
pub const ORT_DYLIB_ENV: &str = "ORT_DYLIB_PATH";

/// Where the Docker images put the library.
pub const ORT_DYLIB_DEFAULT: &str = "/opt/onnxruntime/lib/libonnxruntime.so";

/// One model `pipes` can run: the published network, as a file derived from
/// it with the declared input at the size the pipeline feeds, and that size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSpec {
    /// The name `--detector-model` takes and `run.json` records.
    pub name: &'static str,
    /// The derived file, relative to the working directory (the repository
    /// root, where `runs/` also goes).
    pub file: &'static str,
    /// sha256 of that file: the one file this spec will load.
    pub sha256: &'static str,
    /// The published file it is derived from, in `models/`.
    pub source: &'static str,
    /// That file's sha256.
    pub source_sha256: &'static str,
    /// Network input width and height in pixels.
    ///
    /// 640x192 is the KITTI frame's aspect ratio (1242/375 = 3.31) rounded to
    /// the network's stride of 32: r = 0.512, content 635x192, five columns
    /// of padding. The published 416x416 would spend 70 % of its compute on
    /// padding and see objects at the scale of a 416x128 input. 1024x320 is
    /// the same ratio at r = 0.824: four fifths of the frame's own
    /// resolution, for the models with the capacity to use it. 1248x384 is
    /// the frame itself, to the stride: r = 1.005, content 1248x376.
    pub input_w: usize,
    /// Network input height in pixels. See `input_w`.
    pub input_h: usize,
    /// Parameters, millions, from the model zoo: the size of the network.
    pub params_m: f32,
    /// Whether the file's weights and inner tensors are 16-bit floats, its
    /// input and output still 32-bit: a file for ONNX Runtime, and in
    /// practice for the GPU. tract is not given one.
    pub half: bool,
}

/// YOLOX-Nano at 640x192: the detector every run before sprint 2 used, and
/// still the default. 0.9 M parameters.
pub const YOLOX_NANO: ModelSpec = ModelSpec {
    name: "yolox_nano",
    file: "models/yolox_nano-640x192.onnx",
    sha256: "2e657f4a4dd9673fe8952d30290c4ef522ae0898ce5a834f6187f320e502efde",
    source: "yolox_nano.onnx",
    source_sha256: "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    input_w: 640,
    input_h: 192,
    params_m: 0.91,
    half: false,
};

/// YOLOX-S at 640x192: ten times the parameters of Nano at the same input.
pub const YOLOX_S: ModelSpec = ModelSpec {
    name: "yolox_s",
    file: "models/yolox_s-640x192.onnx",
    sha256: "e299a91106dff16bf0c7150f734ca66554d613ea862f90d0486d64c4e3ad7db8",
    source: "yolox_s.onnx",
    source_sha256: "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    input_w: 640,
    input_h: 192,
    params_m: 9.0,
    half: false,
};

/// YOLOX-S at 1024x320: the same network given four fifths of the frame's
/// resolution.
pub const YOLOX_S_1024: ModelSpec = ModelSpec {
    name: "yolox_s-1024",
    file: "models/yolox_s-1024x320.onnx",
    sha256: "78bca087b51cb8c0858d136d06c81cbb823f76706933ecc834da40f6dbbf04dc",
    source: "yolox_s.onnx",
    source_sha256: "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    input_w: 1024,
    input_h: 320,
    params_m: 9.0,
    half: false,
};

/// YOLOX-M at 1024x320: 25 million parameters.
pub const YOLOX_M_1024: ModelSpec = ModelSpec {
    name: "yolox_m-1024",
    file: "models/yolox_m-1024x320.onnx",
    sha256: "44c08f3bf3623416823cc975d9d4018471d1eac1174eb16e944b6940659a1867",
    source: "yolox_m.onnx",
    source_sha256: "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f",
    input_w: 1024,
    input_h: 320,
    params_m: 25.3,
    half: false,
};

/// YOLOX-L at 1248x384, 16-bit weights: 54 million parameters given the
/// frame at its own resolution. For the GPU.
pub const YOLOX_L_1248: ModelSpec = ModelSpec {
    name: "yolox_l-1248",
    file: "models/yolox_l-1248x384-fp16.onnx",
    sha256: "a12d84e874f8a0a28e918708378da6bfc2f73f113c4be5d7741ace980818ac0d",
    source: "yolox_l.onnx",
    source_sha256: "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
    input_w: 1248,
    input_h: 384,
    params_m: 54.2,
    half: true,
};

/// YOLOX-X at 1248x384, 16-bit weights: 99 million parameters, the biggest
/// network of the family, at the frame's own resolution. For the GPU.
pub const YOLOX_X_1248: ModelSpec = ModelSpec {
    name: "yolox_x-1248",
    file: "models/yolox_x-1248x384-fp16.onnx",
    sha256: "8a0b683a887c3cf7242f123dfecf4644e7aafc3dab6dfd80f6022667591d0169",
    source: "yolox_x.onnx",
    source_sha256: "c892d7aaf1c4746d8a4d675bec669a4db4f434b4ee1efb654bc9b353379c7c55",
    input_w: 1248,
    input_h: 384,
    params_m: 99.1,
    half: true,
};

/// Every model `--detector-model` can name, the default first.
pub const MODELS: [&ModelSpec; 6] = [
    &YOLOX_NANO,
    &YOLOX_S,
    &YOLOX_S_1024,
    &YOLOX_M_1024,
    &YOLOX_L_1248,
    &YOLOX_X_1248,
];

/// The model of that name, or `None`.
pub fn model_named(name: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().copied().find(|m| m.name == name)
}

/// Where `pipes run` looks for the default detector's weights, relative to
/// the working directory: [`YOLOX_NANO`]'s file.
pub const MODEL_FILE: &str = YOLOX_NANO.file;

/// The script that puts the models there (`scripts/prepare_models.py` does
/// the deriving; this one downloads and runs it on Windows). Named by every
/// error that means "the model is not usable", so the message ends at a
/// next step.
pub const FETCH_SCRIPT: &str = "scripts/fetch_model.ps1";

/// sha256 of the default detector's file: [`YOLOX_NANO`]'s.
pub const MODEL_SHA256: &str = YOLOX_NANO.sha256;

/// The default model's name, as `run.json` and the schema metadata record it.
pub const MODEL_NAME: &str = YOLOX_NANO.name;

/// The default model's input width in pixels. See [`ModelSpec::input_w`].
pub const INPUT_W: usize = YOLOX_NANO.input_w;
/// The default model's input height in pixels.
pub const INPUT_H: usize = YOLOX_NANO.input_h;

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

/// Anchors (output rows) of the default model, at [`INPUT_W`]x[`INPUT_H`]:
/// 1920 + 480 + 120. See [`ModelSpec::anchors`].
pub const ANCHORS: usize = YOLOX_NANO.anchors();

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
        /// What the spec pins.
        expected: &'static str,
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
            "run `docker compose build` to put it in the image (outside Docker, \
             `powershell -ExecutionPolicy Bypass -File {}` from the repository root), \
             or pass `--detector off`",
            FETCH_SCRIPT.replace('/', "\\")
        );
        match self {
            CamDetError::Missing(p) => write!(
                f,
                "the camera detector's model is missing: {} does not exist; {fetch}",
                p.display()
            ),
            CamDetError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            CamDetError::Sha256Mismatch {
                path,
                got,
                expected,
            } => write!(
                f,
                "{} is not the pinned camera detector: sha256 {got}, expected {expected}; \
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

/// The network as one runtime holds it.
enum Engine {
    /// tract's optimised plan: `run` takes `&self` and makes its own state.
    Tract(Arc<TypedRunnableModel>),
    /// An ONNX Runtime session, whose `run` takes `&mut self`: one per
    /// stage, so two cameras never queue on one lock.
    Ort(Mutex<ort::session::Session>),
}

/// The loaded network. Built once, before the clock starts; `infer` is the
/// only thing a frame costs.
pub struct Model {
    engine: Engine,
    /// Which model, and so which input size and which file.
    pub spec: ModelSpec,
    /// Which runtime runs it.
    pub backend: Backend,
    /// The CPU threads the `ort` backend was given; 1 for the others.
    pub threads: usize,
    /// sha256 of the file it was loaded from — equal to the spec's, or
    /// [`Model::load_with`] would have refused it.
    pub sha256: String,
}

impl Model {
    /// The default: [`YOLOX_NANO`] on tract, from `path`.
    pub fn load(path: &Path) -> Result<Model, CamDetError> {
        Model::load_with(path, &YOLOX_NANO, Backend::Tract, 1, CudaOptions::default())
    }

    /// Reads `path`, checks its sha256 against `spec`'s, and builds the
    /// network on `backend` at 1x3x`input_h`x`input_w`; `threads` is the
    /// `ort` backend's CPU pool and `cuda` the `cuda` backend's knobs, each
    /// ignored by the other backends.
    ///
    /// On tract the file's declared shapes are dropped (`with_ignore_value_info`
    /// and `with_ignore_output_shapes`) and the input fact set to the spec's
    /// size: any multiple of 32 works, the network's reshapes being `[1, 85,
    /// -1]`. ONNX Runtime takes the declared shapes as they are, which is
    /// why every spec's file is the published graph with its declared input
    /// at the spec's size.
    pub fn load_with(
        path: &Path,
        spec: &ModelSpec,
        backend: Backend,
        threads: usize,
        cuda: CudaOptions,
    ) -> Result<Model, CamDetError> {
        // Refused by name, before any file is opened: a CPU run that asks
        // for a GPU model is told so whether or not the file is there.
        if spec.half && backend == Backend::Tract {
            return Err(CamDetError::Runtime(format!(
                "{} is a file of 16-bit weights, made for ONNX Runtime on the GPU: run it with \
                 --detector-backend cuda (or ort, slowly); tract runs the 32-bit models",
                spec.name
            )));
        }
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
        if sha256 != spec.sha256 {
            return Err(CamDetError::Sha256Mismatch {
                path: path.to_path_buf(),
                got: sha256,
                expected: spec.sha256,
            });
        }
        let engine = match backend {
            Backend::Tract => {
                let rt =
                    |e: tract_onnx::prelude::TractError| CamDetError::Runtime(format!("{e:#}"));
                let plan = tract_onnx::onnx()
                    .with_ignore_value_info(true)
                    .with_ignore_output_shapes(true)
                    .model_for_read(&mut bytes.as_slice())
                    .map_err(rt)?
                    .with_input_fact(0, f32::fact([1, 3, spec.input_h, spec.input_w]).into())
                    .map_err(rt)?
                    .into_optimized()
                    .map_err(rt)?
                    .into_runnable()
                    .map_err(rt)?;
                Engine::Tract(plan)
            }
            // The session the lidar detector gets too, built in one place
            // (`crate::onnx`): the library, the thread pool, cuDNN's
            // algorithm rule and TF32 on the same terms for both networks.
            Backend::Ort | Backend::Cuda => Engine::Ort(Mutex::new(
                ort_session(&bytes, threads, (backend == Backend::Cuda).then_some(cuda))
                    .map_err(CamDetError::Runtime)?,
            )),
        };
        let model = Model {
            engine,
            spec: *spec,
            backend,
            threads: if backend == Backend::Ort {
                threads.max(1)
            } else {
                1
            },
            sha256,
        };
        // The first frame's cost, paid here, before the clock: a runtime
        // allocates its buffers and loads its kernels on its first run --
        // a third of a second on the GPU, where the pipeline's first three
        // frames were then late and the fusion declared the camera silent.
        let blank = vec![0.0f32; 3 * spec.input_w * spec.input_h];
        model.infer(blank, 1.0, spec.input_w as u32, spec.input_h as u32)?;
        Ok(model)
    }

    /// The network input for one RGB8 frame, at this model's size:
    /// [`ModelSpec::preprocess`].
    pub fn preprocess(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<f32>, f64), CamDetError> {
        self.spec.preprocess(rgb, width, height)
    }

    /// Runs the network on one tensor from [`Model::preprocess`], which it
    /// consumes, and decodes the `[anchors x 85]` output **where the runtime
    /// left it**: the output is read in place, never copied out.
    ///
    /// `r`, `width` and `height` are the letterbox ratio `preprocess`
    /// returned and the frame's own size, which the boxes are mapped back to.
    /// The candidates come back in output-row order, before [`nms`].
    pub fn infer(
        &self,
        input: Vec<f32>,
        r: f64,
        width: u32,
        height: u32,
    ) -> Result<Vec<Det>, CamDetError> {
        let (h, w) = (self.spec.input_h, self.spec.input_w);
        let want = self.spec.anchors() * OUTPUT_COLS;
        let wrong = |n: usize| {
            CamDetError::Runtime(format!(
                "output has {n} values, expected {} x {OUTPUT_COLS}",
                self.spec.anchors()
            ))
        };
        match &self.engine {
            Engine::Tract(plan) => {
                let rt =
                    |e: tract_onnx::prelude::TractError| CamDetError::Runtime(format!("{e:#}"));
                // Moved, not copied: the tensor takes the Vec's buffer.
                let t: Tensor = tract_ndarray::Array4::from_shape_vec((1, 3, h, w), input)
                    .map_err(|e| CamDetError::Runtime(e.to_string()))?
                    .into();
                let out = plan.run(tvec!(t.into())).map_err(rt)?;
                let first = out
                    .first()
                    .ok_or_else(|| CamDetError::Runtime("no output".to_string()))?;
                let view = first.to_plain_array_view::<f32>().map_err(rt)?;
                // A tract tensor owns one contiguous, standard-layout buffer,
                // so the view is a slice and the fallback copy is never
                // expected to run; it exists so a layout change would cost
                // time rather than correctness.
                let copied: Vec<f32>;
                let raw: &[f32] = match view.as_slice() {
                    Some(s) => s,
                    None => {
                        copied = view.iter().copied().collect();
                        &copied
                    }
                };
                if raw.len() != want {
                    return Err(wrong(raw.len()));
                }
                Ok(self.spec.decode(raw, r, width, height))
            }
            Engine::Ort(session) => {
                let rt = |e: ort::Error| CamDetError::Runtime(format!("ONNX Runtime: {e}"));
                // Moved, not copied, into the input tensor; the output is
                // read where the runtime left it.
                let t = ort::value::Tensor::<f32>::from_array(([1usize, 3, h, w], input))
                    .map_err(rt)?;
                let mut s = session.lock().unwrap_or_else(PoisonError::into_inner);
                let out = s.run(ort::inputs![t]).map_err(rt)?;
                let (_, raw) = out[0].try_extract_tensor::<f32>().map_err(rt)?;
                if raw.len() != want {
                    return Err(wrong(raw.len()));
                }
                Ok(self.spec.decode(raw, r, width, height))
            }
        }
    }
}

impl ModelSpec {
    /// Anchors (output rows) at this input size: one per cell of each of the
    /// three heads' grids.
    pub const fn anchors(&self) -> usize {
        (self.input_w / 8) * (self.input_h / 8)
            + (self.input_w / 16) * (self.input_h / 16)
            + (self.input_w / 32) * (self.input_h / 32)
    }

    /// The input shape as the words `1x3xHxW`, and that the weights are
    /// 16-bit floats when they are.
    pub fn input_words(&self) -> String {
        let half = if self.half { ", 16-bit weights" } else { "" };
        format!("1x3x{}x{}{half}", self.input_h, self.input_w)
    }

    /// The letterbox scale for a `width x height` frame: the largest `r`
    /// that fits the frame inside the input, as YOLOX computes it (f64, then
    /// truncated).
    pub fn letterbox_ratio(&self, width: u32, height: u32) -> f64 {
        (self.input_h as f64 / f64::from(height)).min(self.input_w as f64 / f64::from(width))
    }

    /// The resized content size inside the letterbox: `trunc(w r) x trunc(h
    /// r)`.
    pub fn letterbox_size(&self, width: u32, height: u32) -> (u32, u32) {
        let r = self.letterbox_ratio(width, height);
        (
            ((f64::from(width) * r) as u32).clamp(1, self.input_w as u32),
            ((f64::from(height) * r) as u32).clamp(1, self.input_h as u32),
        )
    }

    /// The network input for one RGB8 frame, read **in place** from `rgb`.
    ///
    /// Resize bilinear (`Triangle`, which reproduces the reference cv2
    /// `INTER_LINEAR` detections to within 0.01 of score) to
    /// [`ModelSpec::letterbox_size`], place it at the TOP-LEFT of an
    /// `input_w` x `input_h` canvas filled with [`PAD_VALUE`], planes in B,
    /// G, R order, raw 0..255 values, NCHW. Returns the tensor and the ratio
    /// [`ModelSpec::decode`] divides by.
    ///
    /// Three allocations, all real and all reported by the stage: the
    /// resize's own intermediate (its vertical pass writes a four-channel f32
    /// image of the frame's width by the new height, 1242 x 192 x 16 =
    /// 3,815,424 bytes on KITTI at 640x192), the resized image (`trunc(w r) *
    /// trunc(h r) * 3` = 365,760 bytes) and the tensor (`3 * 192 * 640 * 4` =
    /// 1,474,560 bytes) -- 5,655,744 bytes. A run on drive_0005 measures
    /// 5,655,808 per frame: 64 bytes more, not traced.
    pub fn preprocess(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<f32>, f64), CamDetError> {
        let bad = CamDetError::BadFrame {
            width,
            height,
            got: rgb.len(),
        };
        if width == 0 || height == 0 || rgb.len() != width as usize * height as usize * 3 {
            return Err(bad);
        }
        let src: ImageBuffer<Rgb<u8>, &[u8]> =
            ImageBuffer::from_raw(width, height, rgb).ok_or(bad)?;
        let r = self.letterbox_ratio(width, height);
        let (nw, nh) = self.letterbox_size(width, height);
        let resized = image::imageops::resize(&src, nw, nh, image::imageops::FilterType::Triangle);
        let plane = self.input_w * self.input_h;
        let mut t = vec![PAD_VALUE; 3 * plane];
        let (nw, nh) = (nw as usize, nh as usize);
        let raw = resized.as_raw();
        for y in 0..nh {
            let row = &raw[y * nw * 3..(y + 1) * nw * 3];
            let (px, _) = row.as_chunks::<3>();
            let base = y * self.input_w;
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
    /// The output is `[anchors x 85]`, stride-8 cells first (row-major), then
    /// stride 16, then 32. Columns 0..3 are grid-relative `(dx, dy, log w,
    /// log h)`; 4 is objectness and 5..84 the class probabilities, both
    /// already through a sigmoid. `cx = (dx + gx) s`, `w = exp(log w) s`;
    /// corners divided by `r` and clipped to the image. Class = argmax (the
    /// first on a tie).
    pub fn decode(&self, raw: &[f32], r: f64, width: u32, height: u32) -> Vec<Det> {
        let mut out = Vec::new();
        let (wf, hf) = (width as f32, height as f32);
        let r = r as f32;
        let mut row = 0usize;
        for s in STRIDES {
            let (gw, gh) = (self.input_w / s, self.input_h / s);
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
}

/// [`ModelSpec::letterbox_ratio`] of the default model.
pub fn letterbox_ratio(width: u32, height: u32) -> f64 {
    YOLOX_NANO.letterbox_ratio(width, height)
}

/// [`ModelSpec::letterbox_size`] of the default model.
pub fn letterbox_size(width: u32, height: u32) -> (u32, u32) {
    YOLOX_NANO.letterbox_size(width, height)
}

/// [`ModelSpec::preprocess`] of the default model.
pub fn preprocess(rgb: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, f64), CamDetError> {
    YOLOX_NANO.preprocess(rgb, width, height)
}

/// [`ModelSpec::decode`] of the default model.
pub fn decode(raw: &[f32], r: f64, width: u32, height: u32) -> Vec<Det> {
    YOLOX_NANO.decode(raw, r, width, height)
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
    let (t, r) = model.preprocess(rgb, width, height)?;
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
/// are the first two columns; the sample's `parent` names it too. This is
/// the default model's schema; [`cam_det_schema_of`] is any model's.
pub fn cam_det_schema(model_sha256: &str) -> Arc<Schema> {
    cam_det_schema_of(&YOLOX_NANO, model_sha256)
}

/// [`cam_det_schema`] for the model that made the detections: its name and
/// its input size are in the metadata, so a file of detections says which
/// network they are from.
pub fn cam_det_schema_of(spec: &ModelSpec, model_sha256: &str) -> Arc<Schema> {
    let meta = [
        ("model", spec.name.to_string()),
        ("model_sha256", model_sha256.to_string()),
        (
            "input",
            format!("{} bgr letterbox-top-left", spec.input_words()),
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

    /// The catalog: six models, each its own file and hash, the default
    /// first and the three sizes the pipeline feeds, with the anchor counts
    /// those sizes give.
    #[test]
    fn the_catalog_names_six_models_at_three_sizes() {
        assert_eq!(MODELS[0].name, MODEL_NAME);
        assert_eq!(
            (MODEL_FILE, MODEL_SHA256),
            (YOLOX_NANO.file, YOLOX_NANO.sha256)
        );
        let mut files: Vec<&str> = MODELS.iter().map(|m| m.file).collect();
        files.sort_unstable();
        files.dedup();
        assert_eq!(files.len(), MODELS.len(), "two models share a file");
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{}", m.name);
            assert!(
                m.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{}",
                m.name
            );
            assert_eq!(m.source_sha256.len(), 64, "{}", m.name);
            assert!(m.input_w % 32 == 0 && m.input_h % 32 == 0, "{}", m.name);
            assert_eq!(model_named(m.name), Some(m));
            // The derived file's name says its size, and that its weights
            // are 16-bit when they are.
            let tail = if m.half { "-fp16" } else { "" };
            assert!(
                m.file
                    .ends_with(&format!("-{}x{}{tail}.onnx", m.input_w, m.input_h)),
                "{}",
                m.file
            );
            assert_eq!(m.input_words().contains("16-bit"), m.half, "{}", m.name);
        }
        assert_eq!(YOLOX_NANO.anchors(), 2520);
        assert_eq!(YOLOX_S.anchors(), 2520);
        assert_eq!(YOLOX_S_1024.anchors(), 6720);
        assert_eq!(YOLOX_M_1024.anchors(), 6720);
        assert_eq!(YOLOX_L_1248.anchors(), 9828);
        assert_eq!(YOLOX_X_1248.anchors(), 9828);
        assert_eq!(model_named("yolox_l"), None);
        // The biggest input is the frame itself, to the network's stride:
        // six columns and eight rows of padding, and no resolution given up.
        assert_eq!(YOLOX_X_1248.letterbox_size(1242, 375), (1248, 376));
        assert!((YOLOX_X_1248.letterbox_ratio(1242, 375) - 1.0048).abs() < 1e-3);
        assert_eq!(YOLOX_X_1248.input_words(), "1x3x384x1248, 16-bit weights");
        // The 16-bit files are ONNX Runtime's: tract is refused them by
        // name, before any file is opened for it.
        assert!(MODELS.iter().filter(|m| m.half).count() == 2);
        let refused = Model::load_with(
            Path::new("no/such/file.onnx"),
            &YOLOX_X_1248,
            Backend::Tract,
            1,
            CudaOptions::default(),
        );
        assert!(
            matches!(&refused, Err(CamDetError::Runtime(m)) if m.contains("16-bit weights")),
            "a GPU model on tract was not refused by name"
        );
        // The bigger input keeps the frame's aspect ratio at four fifths of
        // its resolution.
        assert_eq!(YOLOX_S_1024.letterbox_size(1242, 375), (1024, 309));
        assert!((YOLOX_S_1024.letterbox_ratio(1242, 375) - 0.8245).abs() < 1e-3);
        assert_eq!(YOLOX_S_1024.input_words(), "1x3x320x1024");
        for b in [Backend::Tract, Backend::Ort, Backend::Cuda] {
            assert_eq!(b.is_ort(), b != Backend::Tract);
        }
    }

    /// Decoding at the bigger size: the grids are 128x40, 64x20 and 32x10,
    /// and an anchor of the second head lands where its cell says.
    #[test]
    fn decode_at_1024_by_320_reads_the_bigger_grids() {
        let spec = YOLOX_S_1024;
        let mut raw = vec![0f32; spec.anchors() * OUTPUT_COLS];
        // Row 5120 + 3 * 64 + 7: stride 16, cell (7, 3).
        let i = 128 * 40 + 3 * 64 + 7;
        let a = &mut raw[i * OUTPUT_COLS..(i + 1) * OUTPUT_COLS];
        a[0] = 0.5;
        a[1] = 0.5;
        a[2] = 0.0;
        a[3] = 0.0;
        a[4] = 1.0;
        a[5 + 2] = 0.9;
        let d = spec.decode(&raw, 0.8245, 1242, 375);
        assert_eq!(d.len(), 1);
        let d = d[0];
        assert_eq!((d.class_id, d.row), (2, i as u32));
        // Centre (7.5, 3.5) cells of 16 px, a 16 px box, over the ratio.
        let (cx, cy) = (7.5 * 16.0 / 0.8245, 3.5 * 16.0 / 0.8245);
        assert!((0.5 * (d.x0 + d.x1) - cx).abs() < 0.01, "{d:?}");
        assert!((0.5 * (d.y0 + d.y1) - cy).abs() < 0.01, "{d:?}");
        assert!((d.x1 - d.x0 - 16.0 / 0.8245).abs() < 0.01, "{d:?}");
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
        assert_eq!(schema.metadata()["model"], MODEL_NAME);
        assert_eq!(
            schema.metadata()["input"],
            "1x3x192x640 bgr letterbox-top-left"
        );
        // Another model's batches name that model, not the default.
        let big = cam_det_schema_of(&YOLOX_L_1248, YOLOX_L_1248.sha256);
        assert_eq!(big.metadata()["model"], "yolox_l-1248");
        assert_eq!(
            big.metadata()["input"],
            "1x3x384x1248, 16-bit weights bgr letterbox-top-left"
        );
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
