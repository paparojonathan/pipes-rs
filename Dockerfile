# syntax=docker/dockerfile:1.7
#
# pipes with nothing installed but Docker: the build fetches the Rust
# toolchain, compiles the binary and downloads the detector; the `fetch`
# target downloads KITTI drives; the `test` target runs the tests.
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

# ---- Model: the frozen detector ---------------------------------------------
# The URL and SHA-256 scripts/fetch_model.ps1 pins; the build stops if the
# download does not match, and the binary checks the hash again at load.
FROM debian:bookworm-slim AS model
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
ARG MODEL_URL=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
ARG MODEL_SHA256=c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
RUN mkdir /models \
 && curl -fsSL --retry 3 -o /models/yolox_nano.onnx "$MODEL_URL" \
 && echo "$MODEL_SHA256  /models/yolox_nano.onnx" | sha256sum -c -
COPY <<'EOF' /models/THIRD_PARTY.md
# Third-party files in models/

Downloaded when the Docker image is built (the same file and hash as
`scripts/fetch_model.ps1`), never committed.

## yolox_nano.onnx

- What: YOLOX-Nano, an 80-class COCO object detector, as the ONNX export
  Megvii publishes (opset 11, input `images` 1x3x416x416, output
  `output` 1x3549x85). `pipes` loads it at 1x3x192x640.
- From: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
  (release 0.1.1rc0 of https://github.com/Megvii-BaseDetection/YOLOX)
- sha256: c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
- Size: 3,659,407 bytes.
- Trained on: COCO 2017.
- Licence of the code: Apache-2.0
  (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE).
  The repository has no NOTICE file.
- Licence of the weights: **inferred, not stated.** The release carries no
  separate licence for the ONNX files; they are assets of the same
  repository, linked from its documentation as its pre-generated models, and
  are treated here as covered by the repository's Apache-2.0 grant. That is
  an inference from where they are published, not a quoted statement.
EOF

# ---- Data: `docker compose run --rm fetch [--smaller | DRIVE...]` ------------
# Downloads KITTI raw drives (the synced+rectified zips) and their dates'
# calibration into /data/kitti, skipping what is already there. With no
# argument it fetches the four drives the README uses (3.2 GB); `--smaller`
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
  -h|--help)
    echo "fetch [--smaller | DRIVE...]"
    echo "  no argument  the four drives the README uses, 0005 0009 0013 0048 (3.2 GB)"
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
    continue
  fi
  echo "fetching $drive"
  curl -fL --retry 3 --progress-bar -o "$root/.drive.zip" "$base/${drive%_sync}/$drive.zip"
  rm -rf "$root/.partial"
  unzip -q "$root/.drive.zip" -d "$root/.partial"
  mv "$root/.partial/$date/$drive" "$root/$date/"
  rm -rf "$root/.partial" "$root/.drive.zip"
done
echo "drives in $root:"
ls -d "$root"/*/*_sync
EOF
ENTRYPOINT ["fetch-kitti"]

# ---- Test: `docker compose run --rm test [cargo test args]` ----------------
# The build stage's toolchain and sources, with the detector where the tests
# look for it. compose.yaml mounts the KITTI data and keeps cargo's registry
# and target directory in volumes, so only the first run compiles everything.
FROM build AS test
COPY --from=model /models models
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
