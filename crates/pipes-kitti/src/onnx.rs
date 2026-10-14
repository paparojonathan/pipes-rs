//! ONNX Runtime, shared by the two frozen detectors: loaded once per process
//! from the library [`crate::camdet::ORT_DYLIB_ENV`] names, and a session
//! built the same way for either network, on the CPU or on the GPU.
//!
//! Here rather than in [`crate::camdet`] because the lidar detector
//! ([`crate::pillars`]) needs exactly the same runtime on exactly the same
//! terms -- the library, the thread pool, cuDNN's algorithm rule, TF32 --
//! and two copies of that would be two places for them to drift.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::camdet::{CudaOptions, CudaSearch, ORT_DYLIB_DEFAULT, ORT_DYLIB_ENV};

/// ONNX Runtime, initialised once per process from the library
/// [`ORT_DYLIB_ENV`] names, and refused with a sentence rather than a panic
/// when it is not there.
pub fn ort_ready() -> Result<(), String> {
    static READY: OnceLock<Result<(), String>> = OnceLock::new();
    READY
        .get_or_init(|| {
            let path =
                std::env::var(ORT_DYLIB_ENV).unwrap_or_else(|_| ORT_DYLIB_DEFAULT.to_string());
            if !Path::new(&path).is_file() {
                return Err(format!(
                    "ONNX Runtime is not at {path} ({ORT_DYLIB_ENV}): the `ort` and `cuda` backends need \
                     libonnxruntime.so 1.22.0 there (the GPU build for `cuda`), which the Docker images carry. \
                     The camera detector's default, --detector-backend tract, needs none of it; the lidar \
                     detector has no other runtime"
                ));
            }
            ort::init_from(&path)
                .commit()
                .map(|_| ())
                .map_err(|e| format!("ONNX Runtime at {path} could not be initialised: {e}"))
        })
        .clone()
}

/// The NVIDIA driver's own library, which the CUDA runtime loads the moment a
/// session asks for the GPU.
const CUDA_DRIVER_LIB: &str = "libcuda.so.1";

/// The loader's own directories on Linux, searched after everything
/// `LD_LIBRARY_PATH` and `/etc/ld.so.conf` name.
const LOADER_DIRS: [&str; 6] = [
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/lib64",
    "/usr/lib64",
    "/lib",
    "/usr/lib",
];

/// The directories one `ld.so.conf` file lists: a path a line, comments,
/// blank lines and `include` lines left out.
fn ld_conf_dirs(text: &str) -> Vec<PathBuf> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| l.starts_with('/'))
        .map(PathBuf::from)
        .collect()
}

/// Everywhere the dynamic loader looks for a library by name:
/// `LD_LIBRARY_PATH`, what `/etc/ld.so.conf` and `/etc/ld.so.conf.d/*.conf`
/// list, and its own defaults.
fn loader_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::env::var("LD_LIBRARY_PATH")
        .map(|v| {
            v.split(':')
                .filter(|d| !d.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default();
    let mut conf: Vec<PathBuf> = vec![PathBuf::from("/etc/ld.so.conf")];
    if let Ok(entries) = std::fs::read_dir("/etc/ld.so.conf.d") {
        let mut more: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        // A fixed order, so the same machine gives the same list.
        more.sort();
        conf.extend(more);
    }
    for file in conf {
        if let Ok(text) = std::fs::read_to_string(&file) {
            dirs.extend(ld_conf_dirs(&text));
        }
    }
    dirs.extend(LOADER_DIRS.iter().map(PathBuf::from));
    dirs
}

/// Whether the NVIDIA driver's library is somewhere the loader looks.
///
/// Asked before a CUDA session is built, because without it the failure
/// is not an error. A container that was not handed the GPU has the CUDA
/// runtime and no driver, and under ONNX Runtime 1.22 the provider then
/// ends the process with a segmentation fault instead of returning: a run
/// that prints its calibration and exits 139. Checked on Linux only, where
/// the Docker images run; elsewhere the provider is left to speak for
/// itself.
fn cuda_driver_found() -> bool {
    if !cfg!(target_os = "linux") {
        return true;
    }
    loader_dirs()
        .iter()
        .any(|d| d.join(CUDA_DRIVER_LIB).exists())
}

/// A session over the network in `bytes`: graph optimisations at their
/// highest level, `threads` on the CPU pool, and with `cuda` the CUDA
/// provider -- the algorithms by cuDNN's rule rather than a timing race,
/// unless the options say otherwise, which is what makes two runs on one
/// GPU the same bytes -- refused outright when the provider cannot load.
pub fn ort_session(
    bytes: &[u8],
    threads: usize,
    cuda: Option<CudaOptions>,
) -> Result<ort::session::Session, String> {
    use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch;
    use ort::execution_providers::CUDAExecutionProvider;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;
    ort_ready()?;
    if cuda.is_some() && !cuda_driver_found() {
        return Err(format!(
            "the `cuda` backend needs an NVIDIA GPU handed to this process, and the driver's library \
             ({CUDA_DRIVER_LIB}) is nowhere the loader looks. In Docker, run the GPU image with the GPU: \
             `docker compose run --rm pipes-gpu`, or `docker run --gpus all`. Without a GPU, use \
             --detector-backend ort or tract, and --lidar-backend ort"
        ));
    }
    let rt = |e: ort::Error| format!("ONNX Runtime: {e}");
    let mut builder = Session::builder()
        .map_err(rt)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(rt)?
        .with_intra_threads(threads.max(1))
        .map_err(rt)?;
    if let Some(cuda) = cuda {
        let ep = CUDAExecutionProvider::default()
            .with_conv_algorithm_search(match cuda.search {
                CudaSearch::Heuristic => CuDNNConvAlgorithmSearch::Heuristic,
                CudaSearch::Default => CuDNNConvAlgorithmSearch::Default,
                CudaSearch::Exhaustive => CuDNNConvAlgorithmSearch::Exhaustive,
            })
            .with_tf32(cuda.tf32)
            .build()
            .error_on_failure();
        builder = builder.with_execution_providers([ep]).map_err(rt)?;
    }
    builder.commit_from_memory(bytes).map_err(rt)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn an_ld_so_conf_file_lists_directories_and_nothing_else() {
        let text = "# libc default configuration\n/usr/local/lib\n\n  /usr/local/nvidia/lib64  # the driver\n\
                    include /etc/ld.so.conf.d/*.conf\nnot-a-path\n";
        assert_eq!(
            ld_conf_dirs(text),
            [
                PathBuf::from("/usr/local/lib"),
                PathBuf::from("/usr/local/nvidia/lib64")
            ]
        );
        assert!(ld_conf_dirs("").is_empty());
    }

    #[test]
    fn the_loader_is_searched_in_its_own_directories_last() {
        let dirs = loader_dirs();
        let tail: Vec<PathBuf> = LOADER_DIRS.iter().map(PathBuf::from).collect();
        assert!(dirs.ends_with(&tail), "{dirs:?}");
        // The driver's library is looked for by the name the CUDA runtime
        // asks the loader for.
        assert_eq!(CUDA_DRIVER_LIB, "libcuda.so.1");
    }
}
