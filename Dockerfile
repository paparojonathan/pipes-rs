# syntax=docker/dockerfile:1.7
#
# pipes with nothing installed but Docker: the build fetches the Rust
# toolchain, compiles the binary and downloads the detectors; the `fetch`
# target downloads KITTI drives and their labels; the `test` target runs the
# tests; the `score` target scores a run's detectors against the drive's
# labels.
# compose.yaml wires them together; see the README's "Setup" section.

# ---- Build: the release binary ----------------------------------------------
# Cargo's registry and target directory are cache mounts, so after the first
# build a code change only recompiles the pipes crates.
FROM rust:1.98-bookworm AS build
# A machine with little memory for Docker can have the build killed (exit
# 137); fewer parallel jobs use less: `--build-arg CARGO_BUILD_JOBS=2`.
ARG CARGO_BUILD_JOBS
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin pipes ${CARGO_BUILD_JOBS:+--jobs $CARGO_BUILD_JOBS} \
 && cp target/release/pipes /usr/local/bin/pipes

# ---- Models: the frozen detectors -------------------------------------------
# The ONNX files Megvii publishes for YOLOX-Nano, -S, -M, -L and -X and the
# one NVIDIA publishes for PointPillars, each checked against the sha256
# scripts/prepare_models.py pins, and the files `pipes` loads derived from
# them by that script. A camera model is the same graph with its declared
# input at the size the pipeline feeds (ONNX Runtime insists on it; tract
# ignores it). The lidar model is the same graph with its one TensorRT-only
# node replaced by standard ONNX, so ONNX Runtime can run it. The files the
# GPU loads (YOLOX-L and -X, and PointPillars) are those graphs again with
# their weights as 16-bit floats. The binary checks every derived file's
# hash again at load. The versions are pinned because the bytes depend on
# them: another onnx or numpy writes another file, and the hash refuses it.
FROM python:3.12-slim-bookworm AS model
RUN pip install --no-cache-dir onnx==1.17.0 onnxconverter-common==1.14.0 numpy==2.1.3
WORKDIR /
COPY scripts/prepare_models.py /scripts/prepare_models.py
# YOLOX-L and -X as published are 613 MB together and only their derived
# files are ever loaded, so the two published files are not kept.
RUN mkdir /models && python3 /scripts/prepare_models.py && rm /models/yolox_l.onnx /models/yolox_x.onnx
COPY <<'EOF' /models/THIRD_PARTY.md
# Third-party files in models/

Downloaded and derived when the Docker image is built (the same files and
hashes as `scripts/fetch_model.ps1` and `scripts/prepare_models.py`), never
committed.

## yolox_nano.onnx, yolox_s.onnx, yolox_m.onnx

- What: YOLOX-Nano, YOLOX-S and YOLOX-M, 80-class COCO object detectors, as
  the ONNX exports Megvii publishes (opset 11, input `images` 1x3x416x416,
  output `output` 1x3549x85).
- From: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/
  (release 0.1.1rc0 of https://github.com/Megvii-BaseDetection/YOLOX)
- sha256: c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
  (nano, 3,659,407 bytes),
  c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
  (s, 35,858,002 bytes),
  21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f
  (m, 101,259,744 bytes).
- Trained on: COCO 2017.
- Licence of the code: Apache-2.0
  (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE).
  The repository has no NOTICE file.
- Licence of the weights: **inferred, not stated.** The release carries no
  separate licence for the ONNX files; they are assets of the same
  repository, linked from its documentation as its pre-generated models, and
  are treated here as covered by the repository's Apache-2.0 grant. That is
  an inference from where they are published, not a quoted statement.

## yolox_l.onnx, yolox_x.onnx (downloaded, not kept in the image)

- What: YOLOX-L and YOLOX-X, the two biggest networks of the same family,
  from the same release, under the same terms as the three above.
- sha256: 7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411
  (l, 216,746,733 bytes),
  c892d7aaf1c4746d8a4d675bec669a4db4f434b4ee1efb654bc9b353379c7c55
  (x, 396,142,663 bytes).
- Only the files derived from them are loaded, so the image keeps those and
  deletes these two; `scripts/prepare_models.py` downloads them again.

## yolox_l-1248x384-fp16.onnx, yolox_x-1248x384-fp16.onnx

- What: the two files above with their declared input set to 1x3x384x1248
  and every 32-bit float weight and inner tensor converted to 16-bit, the
  input and the output left as 32-bit. Weights too small for 16 bits are
  rounded to the smallest that fits; nothing else changes. On the four
  drives they find what the 32-bit files find, object for object. These are
  the files `--detector-model yolox_l-1248` and `yolox_x-1248` load, on the
  GPU.
- Made by: scripts/prepare_models.py (onnx 1.17.0, onnxconverter-common
  1.14.0, numpy 2.1.3).
- sha256: pinned in crates/pipes-kitti/src/camdet.rs, one per file.

## yolox_nano-640x192.onnx, yolox_s-640x192.onnx, yolox_s-1024x320.onnx, yolox_m-1024x320.onnx

- What: the files above with their declared input set to 1x3x192x640 or
  1x3x320x1024 and their declared output to the anchor count that size
  gives; no weight or node is changed. These are the files `pipes` loads.
- Made by: scripts/prepare_models.py (onnx 1.17.0).
- sha256: pinned in crates/pipes-kitti/src/camdet.rs, one per file.

## pointpillar.onnx

- What: PointPillars (Lang et al., CVPR 2019), a 3D object detector for
  lidar sweeps with three classes (car, pedestrian, cyclist), as the ONNX
  export NVIDIA publishes for TensorRT. Its README gives the checkpoint it
  was exported from as OpenPCDet's `pointpillar_7728.pth` and its accuracy
  on KITTI's validation set as 77.00 / 52.50 / 62.26 (car / pedestrian /
  cyclist, 3D moderate, R11).
- From: https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars, the file
  `model/pointpillar.onnx` at commit 490a8fb745f1f5575910e55eb9e31a7818387a74
  (stored with Git LFS).
- sha256: b48eada794191301b1255a327542b772152fd9a9364b2c7c184e11e58bcbf517
  (19,342,067 bytes).
- Trained on: the training split of KITTI's 3D object detection benchmark.
- Licence of the code: Apache-2.0, both CUDA-PointPillars
  (https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/blob/main/LICENSE) and
  OpenPCDet (https://github.com/open-mmlab/OpenPCDet/blob/master/LICENSE).
- Licence of the weights: **inferred, not stated.** The file is an asset of
  an Apache-2.0 repository and carries no licence of its own. It was trained
  on KITTI, whose data is published under CC BY-NC-SA 3.0 (non-commercial,
  share-alike). Whether those terms reach a network trained on the data is
  not something either repository says, so this project treats the weights
  as it treats the drives it replays: for research and non-commercial use.

## pointpillars-kitti.onnx

- What: the file above with its `PPScatterPlugin` node, a TensorRT plugin no
  other runtime has, replaced by standard ONNX that does the same thing
  (`ConstantOfShape`, `ScatterND`, `Reshape`, `Transpose`), the number of
  pillars left free, and one attribute repaired (`ReduceMax.keepdims`, a
  list where the standard wants a number). No weight is changed. This is the
  file `pipes run --lidar-detector on --lidar-backend ort` loads, on the CPU.
- Made by: scripts/prepare_models.py (onnx 1.17.0).
- sha256: pinned in crates/pipes-kitti/src/pillars.rs.

## pointpillars-kitti-fp16.onnx

- What: the file above with every 32-bit float weight and inner tensor
  converted to 16-bit, its inputs and outputs left as 32-bit. This is the
  file `pipes run --lidar-detector on` loads on the GPU, the default.
- Made by: scripts/prepare_models.py (onnx 1.17.0, onnxconverter-common
  1.14.0, numpy 2.1.3).
- sha256: pinned in crates/pipes-kitti/src/pillars.rs.
EOF

# ---- ONNX Runtime: the second runtime ---------------------------------------
# Microsoft's release build, GPU flavour: libonnxruntime.so serves
# `--detector-backend ort` on the CPU anywhere, and the CUDA provider beside
# it serves `cuda` where CUDA 12 and cuDNN 9 are (the `gpu` image). Loaded
# at run time from ORT_DYLIB_PATH; nothing links it at build time.
FROM debian:bookworm-slim AS ort
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
ARG ORT_VERSION=1.22.0
ARG ORT_SHA256=2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55
RUN curl -fsSL --retry 3 -o /ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
 && echo "$ORT_SHA256  /ort.tgz" | sha256sum -c - \
 && mkdir -p /opt/onnxruntime/lib /opt/onnxruntime-cpu/lib \
 && tar xzf /ort.tgz -C /opt/onnxruntime --strip-components=1 \
      "onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib" \
      "onnxruntime-linux-x64-gpu-${ORT_VERSION}/LICENSE" \
      "onnxruntime-linux-x64-gpu-${ORT_VERSION}/ThirdPartyNotices.txt" \
 && rm /ort.tgz /opt/onnxruntime/lib/libonnxruntime_providers_tensorrt.so \
 && cp -a /opt/onnxruntime/lib/libonnxruntime.so* /opt/onnxruntime/lib/libonnxruntime_providers_shared.so \
      /opt/onnxruntime/LICENSE /opt/onnxruntime/ThirdPartyNotices.txt /opt/onnxruntime-cpu/ \
 && mv /opt/onnxruntime-cpu/libonnxruntime.so* /opt/onnxruntime-cpu/libonnxruntime_providers_shared.so /opt/onnxruntime-cpu/lib/

# ---- Data: `docker compose run --rm fetch [--list | --smaller | DRIVE...]` ----
# Downloads KITTI raw drives (the synced+rectified zips), their dates'
# calibration and, for the drives KITTI labelled, their labels (the 3D box
# of every road user a person marked, which the pipeline never reads and
# `score` holds its detectors to) into /data/kitti, skipping what is already
# there. With no
# argument it fetches four labelled drives to start with (3.2 GB); `--list`
# names every drive KITTI labelled; `--smaller`
# fetches drive 0005 alone (650 MB), the one a plain run replays. A drive is
# unpacked beside the others and moved into place only when complete, so an
# interrupted download is fetched again rather than left half there.
FROM debian:bookworm-slim AS fetch
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p -m 777 /data/kitti
COPY --chmod=755 <<'EOF' /usr/local/bin/fetch-kitti
#!/bin/sh
set -eu
root=/data/kitti
base=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
all="2011_09_26_drive_0005_sync 2011_09_26_drive_0009_sync 2011_09_26_drive_0013_sync 2011_09_26_drive_0048_sync"
case "${1:-}" in
  "") set -- $all ;;
  --smaller) set -- 2011_09_26_drive_0005_sync ;;
  --list)
    # The drives KITTI labelled: those with a tracklets zip beside them on its
    # server, all of them recorded on 2011_09_26.
    listing=$(curl -fsS --retry 3 "${base%/raw_data}?list-type=2&prefix=raw_data/2011_09_26_drive_")
    echo "the drives KITTI labelled, and the size of each download:"
    printf '%s' "$listing" | tr '<' '\n' | awk '
      /^Key>/ { k = substr($0, 5) }
      /^Size>/ {
        n = split(k, p, "/"); f = p[n]
        if (substr(f, length(f) - 8) == "_sync.zip") size[substr(f, 1, length(f) - 4)] = substr($0, 6)
        if (substr(f, length(f) - 13) == "_tracklets.zip") lab[substr(f, 1, length(f) - 14) "_sync"] = 1
      }
      END { for (d in lab) print "  " d "  " sprintf("%.1f GB", size[d] / 1e9) }' | sort
    exit 0 ;;
  -h|--help)
    echo "fetch [--list | --smaller | DRIVE...]"
    echo "  no argument  four labelled drives to start with, 0005 0009 0013 0048 (3.2 GB)"
    echo "  --list       every drive KITTI labelled, and the size of each download"
    echo "  --smaller    drive 0005 only (650 MB), the one a plain run replays"
    echo "  DRIVE...     named drives, e.g. 2011_09_26_drive_0009_sync"
    exit 0 ;;
esac
mkdir -p "$root"
for drive in "$@"; do
  case "$drive" in
    ????_??_??_drive_????_sync) ;;
    *) echo "not a KITTI drive name or option: $drive (see --help)" >&2; exit 2 ;;
  esac
  date=$(printf '%s' "$drive" | cut -c1-10)
  if [ ! -f "$root/$date/calib_velo_to_cam.txt" ]; then
    echo "fetching the $date calibration"
    curl -fsSL --retry 3 -o "$root/.calib.zip" "$base/${date}_calib.zip"
    unzip -q -o "$root/.calib.zip" -d "$root"
    rm -f "$root/.calib.zip"
  fi
  if [ -d "$root/$date/$drive" ]; then
    echo "$drive: already here"
  else
    echo "fetching $drive"
    curl -fL --retry 3 --progress-bar -o "$root/.drive.zip" "$base/${drive%_sync}/$drive.zip"
    rm -rf "$root/.partial"
    unzip -q "$root/.drive.zip" -d "$root/.partial"
    mv "$root/.partial/$date/$drive" "$root/$date/"
    rm -rf "$root/.partial" "$root/.drive.zip"
  fi
  if [ ! -f "$root/$date/$drive/tracklet_labels.xml" ]; then
    rc=0
    curl -fsSL --retry 3 -o "$root/.labels.zip" "$base/${drive%_sync}/${drive%_sync}_tracklets.zip" || rc=$?
    if [ "$rc" -eq 0 ]; then
      rm -rf "$root/.partial"
      unzip -q "$root/.labels.zip" -d "$root/.partial"
      mv "$root/.partial/$date/$drive/tracklet_labels.xml" "$root/$date/$drive/"
      rm -rf "$root/.partial"
      echo "$drive: labels fetched"
    elif [ "$rc" -eq 22 ]; then
      echo "$drive: KITTI publishes no labels for this drive"
    else
      echo "$drive: labels not fetched (curl exit $rc); run fetch again" >&2
    fi
    rm -f "$root/.labels.zip"
  fi
done
echo "drives in $root:"
ls -d "$root"/*/*_sync
EOF
ENTRYPOINT ["fetch-kitti"]

# ---- Score: `docker compose run --rm score runs/<name> <drive>` -------------
# scripts/score_detectors.py with the two packages it needs: a run's
# detections against the drive's labels. compose.yaml mounts `runs/` and the
# KITTI data, so a drive is named, not located.
FROM python:3.12-slim-bookworm AS score
RUN pip install --no-cache-dir numpy==2.1.3 pyarrow==17.0.0
WORKDIR /work/pipes-rs
COPY scripts/score_detectors.py /usr/local/bin/
ENV PIPES_KITTI_ROOT=/work/data/kitti
ENTRYPOINT ["python3", "/usr/local/bin/score_detectors.py"]

# ---- Test: `docker compose run --rm test [cargo test args]` ----------------
# The build stage's toolchain and sources, with the detector where the tests
# look for it. compose.yaml mounts the KITTI data and keeps cargo's registry
# and target directory in volumes, so only the first run compiles everything.
FROM build AS test
COPY --from=model /models models
# The CPU half of ONNX Runtime, for `--detector-backend ort` and its tests.
COPY --from=ort /opt/onnxruntime-cpu /opt/onnxruntime
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV PIPES_KITTI_ROOT=/data/kitti
ENTRYPOINT ["cargo", "test", "--release", "--locked", "--workspace"]
CMD ["--", "--include-ignored"]

# ---- Runtime: the image `docker compose run pipes` starts -------------------
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'
WORKDIR /work/pipes-rs
COPY --from=build /usr/local/bin/pipes /usr/local/bin/pipes
COPY --from=model /models models
# The CPU half of ONNX Runtime: `--detector-backend ort` works here, `cuda`
# needs the `gpu` image below.
COPY --from=ort /opt/onnxruntime-cpu /opt/onnxruntime
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
# The commit the image was built from, for run.json. A checkout without .git
# (a downloaded zip) records "unknown", as a run outside git does.
RUN --mount=type=bind,target=/ctx \
    if [ -d /ctx/.git ]; then cp -a /ctx/.git .git; fi \
 && mkdir -m 777 runs
ENV PIPES_KITTI_ROOT=/work/data/kitti
# A plain run streams to the Rerun viewer open on the host; compose.yaml
# tells it where the host is (PIPES_RERUN_HOST).
ENTRYPOINT ["pipes"]
CMD ["run"]

# ---- GPU runtime: `docker compose run pipes-gpu` ------------------------------
# The same binary and models on NVIDIA's CUDA 12.6 + cuDNN 9 runtime image,
# with the whole of ONNX Runtime's GPU build, so `--detector-backend cuda`
# and `--lidar-detector on` run their networks on the GPU. Ubuntu 24.04,
# whose glibc is newer than the build image's, so the binary runs unchanged.
# About 3 GB more than the plain image, all of it NVIDIA's libraries and the
# CUDA provider.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS gpu
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'
WORKDIR /work/pipes-rs
COPY --from=build /usr/local/bin/pipes /usr/local/bin/pipes
COPY --from=model /models models
COPY --from=ort /opt/onnxruntime /opt/onnxruntime
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
RUN --mount=type=bind,target=/ctx \
    if [ -d /ctx/.git ]; then cp -a /ctx/.git .git; fi \
 && mkdir -m 777 runs
ENV PIPES_KITTI_ROOT=/work/data/kitti
ENTRYPOINT ["pipes"]
# Everything this image is for, by default: the lidar detector, looking
# ahead and behind, and YOLOX-L on both cameras at the frame's own
# resolution, all three sessions on the GPU with 16-bit weights. YOLOX-L is
# the biggest camera model that was also the best when scored against
# KITTI's labels; YOLOX-X is `--detector-model yolox_x-1248`
# (docs/measurements.md, "The biggest models"). The cameras on the CPU instead:
# `run --lidar-detector on --detector-backend ort`.
CMD ["run", "--lidar-detector", "on", "--detector-backend", "cuda", "--detector-model", "yolox_l-1248"]
