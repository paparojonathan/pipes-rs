# pipes-rs

Pipes is a Rust project for running sensor processing and fusion
in front of ROS 2.

## The architectural bet

Pipes owns the path from sensor drivers through processing and fusion. Sensor
drivers write directly into Pipes-owned Apache Arrow buffers. Pipeline stages
share those buffers without repeatedly copying or serializing the sensor data.
After fusion, Pipes converts the much smaller results into standard ROS 2
messages.

```text
camera / lidar / radar / IMU
              |
              v
     Rust sensor drivers
              |
              v
 Pipes metadata + Arrow buffers
              |
              v
 processing, synchronization, and fusion
              |
              v
 detections / tracks / state / health
              |
              v
       standard ROS 2 topics
```

Significant performance gains are expected because the largest data stays inside
one controlled pipeline:

- a sensor can write into its pipeline buffer once;
- processing, recording, visualization, and fusion can share that data;
- large images and point clouds do not need to be serialized and reconstructed
  between every stage; and
- ROS conversion happens once, after the data has been reduced to the result the
  rest of the robot needs.

Pipes should interoperate cleanly with ROS rather than replace it. ROS can keep
handling control, planning, mission logic, user interfaces, and communication
across the robot. Existing ROS nodes consume normal topics and do not need to
know that Pipes is running the sensor pipeline.

The same pipeline can run from live sensors, simulation, or recorded MCAP data.
Pipes records sensor timing, input order, synchronization decisions, delays, and
drops. This lets developers isolate the sensor and fusion subsystem, replay the
same workload, tune it quickly, and determine whether a bad result came from the
fusion algorithm or from data arriving late or going missing.

## Fusion in Motion

[Fusion in Motion](https://github.com/EthanMBoos/fusion-in-motion) is a separate
simulation playground. It can generate repeatable sensor workloads for Pipes,
but neither project depends on the other.

## This branch

This branch builds the first working slice of that pipeline. It replays a recorded
KITTI drive, camera and lidar together, through Pipes in real time, fuses the two
into an answer about the objects around the car at every lidar sweep, and records
what happened to every sample on the way. What we found on drive 0009, sweep by
sweep, is in the [Sprint 1 findings](https://claude.ai/artifact/3p5HVyjUMuV8PATy3pdqpF).

## Setup

`pipes` runs in Docker, so the Rust toolchain, the build and the detector's
weights all stay inside the container. On your own machine you need only two
things:

1. [Docker](https://docs.docker.com/get-started/get-docker/): Docker Desktop on
   Windows and macOS, Docker Engine on Linux.

2. The Rerun viewer, which shows [the dashboard](#the-dashboard). It runs on your
   machine rather than in the container, because the container has no display:

   ```sh
   python -m pip install rerun-sdk==0.38.1
   ```

   In a new terminal, `rerun --version` should report 0.38.1, matching the Rerun
   library inside `pipes`.

Then, from the `pipes-rs` folder, download the KITTI data once and check that
`pipes` finds it:

```sh
docker compose run --rm fetch          # the four KITTI drives and their calibration (3.2 GB)
docker compose run --rm pipes drives   # each drive with its frame counts and health
```

`fetch --smaller` downloads drive 0005 alone (650 MB), which is all a plain run
needs, and `fetch` followed by drive names downloads just those. Drives already
downloaded are skipped, so running it again only fills in what is missing. The
data lives in a Docker volume named `kitti`.

The first use of each `docker compose` command builds its image, which takes a few
minutes. The detector's weights are downloaded during the build, and the build
stops if their SHA-256 does not match the pinned hash.

## First run

A run uses two terminals: one for the viewer and one for `pipes`. Start them in
this order.

**Terminal 1, the viewer.** Start it first and leave it open:

```sh
rerun
```

**Terminal 2, the runs,** from the `pipes-rs` folder. Each command streams into the
viewer in terminal 1, one run after another:

```sh
docker compose run --rm pipes                                          # drive 0005
docker compose run --rm pipes run --drive 2011_09_26_drive_0009_sync   # then another drive
```

The first command replays drive 0005: 154 camera frames and 154 lidar sweeps over 16
seconds. The run streams to the viewer on your machine (`host.docker.internal:9876`
from inside the container) and waits for it to be ready before it starts the clock;
if no viewer is open, it waits up to a minute and then stops. Windows may ask once
to let the viewer through the firewall. When the drive ends, the terminal prints a
summary and the container exits; the window stays open so you can scrub back through
the run, and later runs stream into the same window. `echo $?` (or `$LASTEXITCODE`
in PowerShell) then says whether the run passed its checks; [Output](#output)
explains the codes. The run's files are in `runs/<name>/` in this checkout.

Any other run, such as the second command above or the
[experiments](#experiments), is `docker compose run --rm pipes run` followed by its
flags, in terminal 2 while the viewer stays open.

- `--rerun rrd` writes `runs/<name>/cam0.rrd` instead, to open later with
  `rerun runs/<name>/cam0.rrd`; no viewer needs to be open during the run.
- To use KITTI data you already have instead of the `kitti` volume, put
  `KITTI_DATA=<path>` in a `.env` file beside `compose.yaml`. The folder must have
  the layout `fetch` produces, for example `KITTI_DATA=../data/kitti` with:

  ```text
  data/kitti/2011_09_26/
    calib_cam_to_cam.txt
    calib_velo_to_cam.txt
    2011_09_26_drive_0005_sync/
      image_02/          timestamps.txt, data/*.png
      velodyne_points/   timestamps.txt, timestamps_start.txt, timestamps_end.txt, data/*.bin
  ```

- A build killed with exit 137 ran out of memory; fewer parallel jobs use less:
  `docker compose build --build-arg CARGO_BUILD_JOBS=2`.
- On Linux the run writes `runs/` as user 1000; if your id differs, set
  `PIPES_UID` and `PIPES_GID` in `.env`.

## What we built

Two sensor drivers, one thread each, read the drive from disk and release each sample
when the real sensor would have delivered it. Both hand their samples to a single
admission, which puts every sample into one arrival order and onto the queue of each
stage that reads it. The stages admit their results the same way, so every stream
follows one set of rules and leaves one kind of record.

Every edge, the connection from one stage to the next, is a bounded queue with a fixed
capacity and a declared policy for when it is full. A full data queue drops its oldest
sample, so a push never waits and a slow stage cannot hold up a sensor; each drop is
recorded with its reason. The detector's queue holds one frame, since a deeper queue
would only deliver older frames. Only the evidence recorder may block, for at most a
second, because losing evidence is worse than a short stall.

The viewer is a consumer like the others. Every picture, the stages' and the
dashboard's, goes onto its own queue of 256 drawings, which one thread hands to Rerun,
so a viewer that freezes loses its oldest drawings, each recorded, and nothing else.

A sample is written once and read in place. The camera driver decodes each PNG into
an Arrow buffer owned by Pipes, the lidar driver loads each sweep into another, and
every later stage reads its input straight out of its producer's buffer. Every
evidence row records the address of the buffer read (`storage_id`) and the bytes the
stage allocated (`bytes_alloc`, from a counting allocator). The same address at both
ends of an edge means the same memory, and the allocations show that reading costs
nothing: a stage allocates only for its own work. The run checks this and prints
lines such as `storage_id equal at 2 stages (driver, rerun): OK`.

The lidar chain has four stages. `reduce` averages the sweep's roughly 120,000 points
into 20 cm cubes, a third of a pedestrian's width, so the smallest object that matters
still spans several cubes. `detect` removes the ground and groups what is left within
30 m into clusters of at least three cubes, each reported as a box. The ground is
fitted by least squares instead of RANSAC, which is random; nothing in the pipeline is
random, so the same input always gives the same output. `track` follows the boxes from
sweep to sweep and gives each an id, a velocity and an age. `state` turns the tracks
into the answer. Per sweep, 1.95 MB of points becomes 492 kB of cubes, 8 kB of boxes,
20 kB of tracks and a 13 kB answer, about 150 times smaller than the sweep.

The camera side is a frozen detector: YOLOX-Nano, pretrained on the 80 everyday object
classes of the COCO dataset. It runs on each camera frame in its own stage, `camdet`,
reading the frame in place. Frozen means the model's SHA-256 is checked at load and
nothing learns. It runs through `tract`, a pure-Rust ONNX runtime, on one thread, so
the same frame always gives the same bytes and two runs can be compared detection by
detection. ONNX Runtime is the usual choice, but it is a C++ library that the build
must download or link, and it brings its own thread pool.

Fusion pairs the sensors by time. A lidar sweep is a 103 ms rotation with a start and
an end, and a camera frame is an instant, so each sweep pairs with the frame whose
instant falls inside its window. When the detector finishes a frame after the lidar
chain finishes its sweep, the fusion waits for the camera, by default for up to one
sweep. Each track's 3D box is then projected into the image with KITTI's
calibration, and a track and a detection match, one to one, when their image boxes
overlap by more than half (intersection over union above ½). Above one half a match
is nearly always unique, and a fixed order settles the rare conflicts the same way on
every run. A fused track takes class and confidence from the camera and distance and
speed from the lidar. A detection that matches no track stays in the answer as
camera-only, without a distance.

The answer lists every track at every sweep, one 68-byte record each: id, distance to
the near face of its box, closing speed, time to contact, bearing, age in seconds,
whether and where it appears in the camera image, whether it is in the car's path,
and class and confidence when fused. The nearest object in the path is flagged.
Tracks outside the camera's view stay in the answer; about three in four are outside it.

Every sample leaves one row per hop in `evidence.csv` with its timing, outcome and
memory figures. When the drive ends, the run checks conservation rules against those
rows: on every edge, delivered plus dropped equals admitted; every sensor frame is
admitted or recorded as missing; every detection is fused or camera-only. A default
run checks 35 such rules, each printed as an `INVARIANT` line, and exits with code 2
if one fails.

Frames missing from the source are recorded, and the replay goes on. In KITTI's
layout, line i+1 of a timestamp file is frame i (frames count from 0). A blank line
means the source has no sample for that frame, and so does every frame after a file
ends early. Each such frame becomes a `Missing` row with reason `absent_in_source` at
its place in the schedule, so the replay keeps the hole and invents nothing. Because
pairing is by time, the sensors line up again right after it. Drive 0009 is a real
case: its lidar has no sweeps for frames 177–180, a 414 ms hole. The run replays the
other 443 sweeps, gives the four camera frames in the hole no answer, restarts the
tracker, and every invariant holds. Any other fault in the layout stops the run before
the clock starts, naming the file and line.

The fusion never uses a late camera frame without saying so. By default, a sweep whose
frame is late or dropped expires: it gets no answer, and its evidence row says why
(`pair_late` or `pair_dropped`). `--pair-stale-ms N` lets the fusion use a frame from
up to N ms before the sweep instead, and labels the result stale. That changes the
answer, because the camera saw each object where it was a frame earlier, so some
tracks lose their class or take a wrong one. The evidence names the camera as the
cause: each stale sweep's `track` row says `reason = stale`, `summary.json` counts
them, and the answer's text says `STALE camera 93 ms`. A test checks that the
lidar half of every track matches the healthy run's, so the difference comes entirely
from the camera.

## Experiments

Each experiment is one command for terminal 2, with the viewer open in terminal 1
(see [First run](#first-run)). Each replays drive 0005 unless it names another.

### The healthy case

```sh
docker compose run --rm pipes run
```

The detector sees all 154 frames, every sweep pairs with its own frame, the fusion
matches 482 tracks to detections over the run, and all 35 invariants hold.

### A slow camera

```sh
docker compose run --rm pipes run --consumer-delay-ms 100 --cap 1 --pair-wait-ms 300
```

The detector sleeps an extra 100 ms per frame, so its one-frame queue drops frames
(38 of 154 on one run; it varies). The fusion expires exactly those sweeps, as
`pair_dropped`, and every invariant holds, because the drops were declared.

### A stale camera

```sh
docker compose run --rm pipes run --consumer-delay-ms 50 --pair-wait-ms 0 --pair-stale-ms 500
```

The detector sleeps an extra 50 ms per frame, so each frame is ready only after its
sweep has been processed, and the fusion, which no longer waits, takes the previous
frame, 93 ms old. Matches fall from 482 to 414, and every answer says
`STALE camera 93 ms`.

### Why the fusion waits

```sh
docker compose run --rm pipes run --consumer-delay-ms 50 --pair-wait-ms 0
```

The detector sleeps an extra 50 ms per frame. No frame is lost, but each arrives
after its sweep has been processed, so all 154 sweeps expire as `pair_late` and
there is no answer.

### A gap in the source

```sh
docker compose run --rm pipes run --drive 2011_09_26_drive_0009_sync
```

Drive 0009 has no lidar for frames 177–180 of 447. The run records the four as
absent, answers the other 443 sweeps and exits 0. It takes 46 seconds.

### Any other drive

```sh
docker compose run --rm pipes drives
docker compose run --rm pipes run --drive 2011_09_26_drive_0013_sync
```

The first command lists the drives; the second replays one of them. Drives 0005,
0009, 0013 and 0048 of 2011_09_26 have camera and lidar. A drive without
`velodyne_points`, such as a camera-only copy of 2011_09_28_drive_0001, replays the
camera alone.

### Without the detector

```sh
docker compose run --rm pipes run --detector off
```

The chain without the camera model: tracks still get image boxes but no class, and
nothing is camera-only.

### A saved recording

```sh
docker compose run --rm pipes run --rerun rrd --name saved
rerun runs/saved/cam0.rrd
```

The first command saves the recording as `runs/saved/cam0.rrd` (about 340 MB for
drive 0005) instead of streaming it, so no viewer needs to be open. The second, on
your machine, reopens it in the same layout.

## Tests

The tests prove the rules in [What we built](#what-we-built). They run in Docker on
the code in this checkout, so an edit needs no rebuild of the image:

```sh
docker compose run --rm test
```

The first run compiles the project, which takes a few minutes; later runs reuse the
build. The tests are built in release mode, because a debug build is too slow for
the real-time tests. Most run `pipes` on small synthetic drives. Ten need the data
from `fetch` and the detector, which is in the test image; they check that runs
repeat, that a stale camera changes only the camera half of each track, that slowing
the detector expires exactly the frames it lost, that drive 0009 records frames
177–180 as absent, and that the reported distances match the raw lidar points on most
sweeps. Before `fetch`, `docker compose run --rm test --` skips those ten and runs
the other 388.

## The dashboard

The window opens straight into a fixed layout: what the pipeline concluded on the
left, and six tabs on the right that show how.

At the top left is the answer as a headline: the nearest object in the car's path,
with its class if the camera saw it, its distance, its time to contact, and how old
the answer is. Below it is the camera image with a box on every tracked object in
view: green where lidar and camera agree, teal for lidar-only, blue for camera-only,
and thick gold for the answer. Under the image is the lidar in 3D, with the cubes, the
tracks and the answer in gold, and the raw sweep on a second tab. A stale pairing dims
the image, and a sweep with no answer puts the reason in the headline, such as
`no answer for frame 178 · no lidar sweep in the source`.

| Tab | What it shows |
|---|---|
| Demo | Time to contact, each camera instant inside its sweep's window, and the tracks in frame by source. |
| Queues | A health lane per queue and for the pairing (green ok, amber skipping or stale, red dropping or expired, grey for a source gap), how full each queue is, and each drop with its reason. |
| Latency | Each measured time against its budget, such as the answer's age, the fusion's wait and the detector's time per frame. |
| Fusion | A stale camera from cause to effect: detector time, pairing outcome, the camera instant against the sweep's window, and the share of lidar tracks in frame that the camera confirmed. |
| Bytes | The pipeline as a graph with each hop's bytes, the byte chain as a table, allocated over carried bytes per edge (0 is zero-copy), and whether each stage read its producer's buffer. |
| Log | Every warning, such as a drop and its reason, above the answer as one line per sweep. |

## Output

The terminal prints about 180 lines; these are the ones to read first:

```text
fusion sets: COMPLETED 154 (...) | DEGRADED 0 (...) | EXPIRED 0 = dropped 0 + late 0 + absent 0 | NEVER REACHED track 0 of 154 sweeps
ANSWER most urgent: sweep 140, 1.79 s to contact -- object 1838 (car 0.89) at 7.56 m, ...
INVARIANT edge=cam0->camdet delivered=154 dropped=0 admitted=154 -> OK
```

`fusion sets` accounts for every sweep: paired with its own frame, paired with a stale
one, expired (split by reason), or never reached the fusion. The exit code is 0 when
every invariant held, 2 when one failed or a flag was mistyped, 3 when a stage
panicked, and 1 for any other error, such as a missing model.

Each run also writes a folder, `runs/<name>/`, named `run-<unix time>` unless you
pass `--name`.

| File | What it holds |
|---|---|
| `evidence.csv` | One row per sample per hop, and one per drawing sent to the viewer (6,163 for drive 0005, 3,391 of them drawings). Key columns: `edge`, `stage` and `seq` (the source frame's number, shared by every hop, so rows join on it); `due_ns`, `arrival_ns`, `queue_wait_ns` and `measurement_age_ns`; `outcome` and `reason`; `storage_id`, `bytes_alloc` and `payload_bytes`. |
| `summary.json` | The terminal's numbers as JSON, for comparing runs: counts and latency percentiles per stage, the detector, the lidar chain, the fusion and the answer. |
| `run.json` | How the run was set up: arguments, git commit, drive, the model's hash and the clock's origin. |
| `events.csv` | Run-level events, such as a gap in the source and the shutdown. |
| `cam_det.arrows` | Every detection: one Arrow batch per camera frame, with box, class and confidence. |
| `fused.arrows` | Every fusion result: one batch per sweep, with its tracks, the camera frame used and that frame's age. |
| `cam0.rrd` | The recording, only with `--rerun rrd`. |
| `_COMPLETE` | Written last. Without it, the run did not finish and the other files may be cut short. |

The `.arrows` files are Arrow IPC streams, which any Arrow library reads (in Python,
`pyarrow.ipc.open_stream`); each exists only when its stage runs.

## Limits

This branch only replays recorded data. There are no live sensor drivers, no clock
synchronization between real sensors, and no ROS 2 output yet. The car's own motion
is not used, because the GPS/IMU data (OXTS) is not read, so speeds are relative to
the car: a parked car ahead closes at the car's own speed, and the tracker cannot
subtract the car's turning. The detector runs on one thread, so when it takes longer
than the camera's 103 ms period it drops frames. The distance is to the near face of
a track's box, so a wide object that only partly overlaps the car's path can read
too near. The grayscale `proc` stage runs only with `--detector off`, where its bare
frame reference, with no detections, is the camera's half of each pair.

## Layout

```text
crates/pipes-core/    clocks, the bounded queue, evidence rows, the counting allocator
crates/pipes-kitti/   KITTI readers (timestamps, camera, lidar, calibration) and the stages:
                      voxels, detection, tracking, fusion, the answer, the camera detector
crates/pipes/         the pipes program: command line, admission, stage threads, recorder, dashboard
docs/                 Ethan's architecture and literature review
scripts/              fetch_model.ps1, not needed with Docker: the image downloads the weights itself
runs/                 one folder per run (not in git)
Dockerfile            the images: build, the detector's weights, the `fetch` data downloader, the tests
compose.yaml          `docker compose run` for `pipes`, `fetch` and `test`
.github/workflows/    CI: formatting, lints and the tests that need no data
```

## Design

Ethan Boos's [architecture and literature review](docs/architecture-and-literature-review.md)
explains the reasoning behind the architecture and the prior work it builds on. The
code on this branch is MIT-licensed (see `LICENSE`); Ethan's introduction at the top
of this file and his design document are his own and not covered by that licence.
