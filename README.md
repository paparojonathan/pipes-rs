# pipes-rs

pipes-rs replays a recorded drive from the KITTI dataset (two cameras, a lidar and the
GPS/IMU) through a real-time sensor pipeline written in Rust. It tracks everything the
lidar sees, and two kinds of frozen detector name those tracks: YOLOX on each camera,
and PointPillars, the lidar detector, on every lidar sweep. At every sweep it says
what is around the car, what each object is and how far away, and for the nearest
thing in the car's path the time to contact: how long until the car reaches it. A
dashboard shows it live, and every sample's way through the pipeline is recorded and
checked.

## Setup

Everything runs in Docker: the build, every model and the data. You install two
things.

1. [Docker](https://docs.docker.com/get-started/get-docker/).
2. The Rerun viewer, which shows [the dashboard](#the-dashboard). It is the one
   program outside Docker, because a container has no display. It installs with pip,
   so the machine needs Python 3:

   ```sh
   python -m pip install rerun-sdk==0.38.1   # python3 on macOS and Linux
   rerun --version                           # must say 0.38.1, the version built into pipes
   ```

Then, from the `pipes-rs` folder, build for your machine and download the drives:

| Your machine | Build | Then run |
|---|---|---|
| NVIDIA GPU, on Windows (Docker Desktop on WSL 2) or Linux (NVIDIA Container Toolkit) | `docker compose build` | `docker compose run --rm pipes-gpu` |
| No NVIDIA GPU | `docker compose build pipes fetch score test` | `docker compose run --rm pipes` |
| Mac with Apple silicon (not tried) | as without a GPU | `docker compose run --rm pipes`, and nothing that needs ONNX Runtime, which the image carries as an x86-64 build: no `--detector-backend ort` and no lidar detector |

```sh
docker compose run --rm fetch   # four labelled KITTI drives to start with, calibration included (3.2 GB)
```

The build compiles the program and puts every model in the image: six camera models
and the lidar model, 0.8 GB, each checked against a pinned SHA-256. The GPU image adds
about 3 GB of NVIDIA libraries. The drives go into a Docker volume. After pulling new
code, build again.

KITTI labelled 38 of its recorded drives with a 3D box round every road user, and any
of them works: `docker compose run --rm fetch --list` names them with their sizes, and
`fetch <drive> ...` downloads them.

## First run

Two terminals. In the first, start the viewer and leave it open:

```sh
rerun
```

In the second, from the `pipes-rs` folder, start the service your machine runs:

```sh
docker compose run --rm pipes-gpu   # with an NVIDIA GPU
docker compose run --rm pipes       # without one
```

| Service | Cameras | Lidar detector, which names the tracks |
|---|---|---|
| `pipes-gpu` | YOLOX-L on both, on the GPU | PointPillars, on the GPU, looking ahead and behind |
| `pipes` | YOLOX-Nano on both, on the CPU | off: it needs a GPU to keep up |

Either replays the default drive in real time, about 16 seconds of it. The run waits
for the viewer before it starts the clock, and stops after a minute if none is open.
When the drive ends, the terminal prints a summary and the container exits; the
window stays open to scrub back through the run, and later runs stream into it. The
run's files are in `runs/<name>/`.

- Windows may ask once to let the viewer through the firewall.
- The GPU run first prints about 220 red `No execution plans support the graph` lines
  from cuDNN. They do not affect it.
- If the GPU detectors start failing partway through a run (CUDA errors in the
  terminal, sweeps EXPIRED on the `fusion sets` line), run it again. It happened on 2
  of 90 GPU runs here, each time with the viewer window open on the same GPU.
- If Docker cannot hand the container a GPU, `pipes-gpu` does not start. Use `pipes`.
- A build killed with exit 137 ran out of memory:
  `docker compose build --build-arg CARGO_BUILD_JOBS=2 pipes`.
- On Linux the run writes `runs/` as user 1000; if your id differs, set `PIPES_UID`
  and `PIPES_GID` in a `.env` file beside `compose.yaml`.

## The dashboard

Four words come up throughout. A **sweep** is one 103 ms rotation of the lidar. A
**track** is one object followed from sweep to sweep. The **answer** is the pipeline's
result for a sweep: every track, with the nearest one in the car's path flagged. The
**pairing** is the camera frame matched to a sweep; it is stale when that frame is an
older one.

The window opens into a fixed layout. On the left is what the pipeline concluded:

- **Headline.** The nearest object in the car's path: what the cameras call it, what
  the lidar detector calls it, its distance, its time to contact and the answer's
  age. A sweep with no answer shows the reason instead.
- **Camera image**, with a box on every tracked object in view: green where the lidar
  and the cameras agree, teal for lidar only (`?` after a class only one camera gave),
  blue for cameras only, thick gold for the answer. A box's label ends with what the
  lidar detector called it, such as `lidar car 0.95`. The corner names the paired
  frame and what the lidar detector named on the sweep, such as
  `lidar: 2 cars, 1 cyclist`, behind the car included. A dimmed image is a stale
  pairing.
- **Lidar in 3D**: the 20 cm cubes the sweep is reduced to, the tracks, the answer in
  gold, the lidar detector's boxes scoring 0.3 or more in violet, and two thin grey
  lines for the corridor the car is predicted to sweep. The raw sweep is on a second
  tab.

On the right, four tabs show how it got there:

| Tab | What it shows |
|---|---|
| Demo | Time to contact. Under it the pairing: the lidar's sweep as a grey band, the paired camera frame's instant as a blue dot, and a diamond for when each part was done: the camera's detections (blue), `detect`'s boxes (teal), the lidar detector's boxes (violet, on a run that has it) and the answer (gold). Then a zoom onto the sweep's end. A blue or violet diamond above the gold one is a result that came after the answer had gone. |
| Fusion | A health lane for the pairing and for four queues in front of it: the left camera's, the lidar's, `detect`'s and the fusion's (green ok, amber stale or skipping, red expired or dropping, grey for a source gap). The right camera's and the lidar detector's queues have no lane; their losses show in the picture's corner, the Log tab and the terminal. Then the tracks in frame by source, and the share of each sensor's objects in frame that the other confirmed. |
| Pipeline | The pipeline as a graph with each hop's bytes; the byte chain, with what each step built and what reading it cost (0 is zero-copy); and every stage with its median time and what it hands on. |
| Log | Every warning, such as a drop and its reason, above the answer as one line per sweep. |

## Commands

Every command is `docker compose run --rm <service> ...`, typed in the `pipes-rs`
folder.

| Service | What it runs |
|---|---|
| `pipes` | The program, on the CPU. |
| `pipes-gpu` | The program, with the GPU handed to it. |
| `fetch` | The download of the drives: four to start with, `fetch --list` for every drive KITTI labelled, `fetch <drive> ...` for any of them. |
| `score` | `score runs/<name> <drive>` counts how many of the road users KITTI's labellers marked were found by a finished run's detectors. |
| `test` | [The tests](#tests). |

The program has four commands, and `--help` after any of them lists its flags with
their defaults:

```sh
docker compose run --rm pipes --help                  # the four commands
docker compose run --rm pipes drives                  # the drives that are downloaded, and their health
docker compose run --rm pipes run --help              # replay a drive: the command behind every run
docker compose run --rm pipes bench-detector --help   # time one camera model on one runtime
docker compose run --rm pipes bench-lidar --help      # time the lidar detector
```

`pipes-gpu` with nothing after it runs `run --lidar-detector on --detector-backend
cuda --detector-model yolox_l-1248`. Anything typed after `pipes-gpu` replaces all
three, and the program's own defaults are the CPU ones. So every GPU run with flags
starts with them, and a run's first lines name what it loaded (`camdet model ...
yolox_l-1248 ... on cuda`, `pillars model ...`):

```sh
docker compose run --rm pipes-gpu run --lidar-detector on --detector-backend cuda --detector-model yolox_l-1248 <more flags>
```

The flags of `run` you are likely to use:

| Flag | What it does | Default |
|---|---|---|
| `--drive <drive>` | The drive to replay: any `fetch` downloaded (`pipes drives` lists them). | the first one `fetch` downloads |
| `--name <name>` | The run's folder under `runs/`. | `run-<unix time>` |
| `--detector-model <model>` | The camera model, one of [the six](#models-and-drives). | `yolox_nano` |
| `--detector-backend <runtime>` | Where the camera model runs: `tract` (pure Rust, one CPU thread), `ort` (ONNX Runtime on the CPU) or `cuda` (the GPU). | `tract` |
| `--lidar-detector on` | Runs PointPillars on every sweep and names the tracks from it. | `off` |
| `--lidar-backend ort` | Runs it on the CPU, where it cannot keep up with the sweeps. | `cuda` |
| `--lidar-views ahead` | The lidar detector looks ahead only, and answers sooner. | both ways on the GPU |
| `--right-camera off` | The left camera alone, with nothing to confirm its classes or stand in for a lost frame. | on when the drive has one |
| `--detector off` | No camera models: tracks get image boxes but no class. | on |
| `--oxts off` | No GPS/IMU: velocities relative to the car, and a straight path. | on when the drive has it |
| `--rerun rrd` | Saves `runs/<name>/cam0.rrd` instead of streaming; `rerun runs/<name>/cam0.rrd` opens it later. | streams to the viewer |
| `--consumer-delay-ms <n>` | Makes the left camera's detector sleep n ms a frame, to see what a slow camera (100) or a dead one (2000) costs. | 0 |
| `--pair-wait-ms <n>` | How long the fusion waits for a late frame. | one sweep, about 103 ms |
| `--pair-stale-ms <n>` | Lets the fusion use a frame up to n ms old, labelled stale. | 0 |

`--lidar-views` and `--oxts` work but are not listed by `--help`. Some whole lines:

```sh
docker compose run --rm pipes-gpu run --lidar-detector on --detector-backend cuda --detector-model yolox_l-1248 --drive <drive>   # another drive
docker compose run --rm pipes-gpu run --lidar-detector on --detector-backend cuda --detector-model yolox_x-1248   # another camera model
docker compose run --rm pipes-gpu run --lidar-detector on --detector-backend cuda --detector-model yolox_l-1248 --consumer-delay-ms 2000   # a dead camera
docker compose run --rm pipes run --drive <drive> --name mine   # a named run on the CPU
docker compose run --rm score runs/mine <drive>                 # then scored against KITTI's labels
```

To see what your machine can carry, time a model outside the pipeline:

```sh
docker compose run --rm pipes-gpu bench-detector --model yolox_l-1248 --backend cuda   # ms a frame for a camera model on the GPU
docker compose run --rm pipes-gpu bench-lidar                                          # ms a sweep for the lidar detector
docker compose run --rm pipes bench-detector --model yolox_s --backend ort             # a camera model on the CPU
```

In a run, the `camdet ran on ... frames` line says whether the model kept up with every frame.

## Models and drives

Every model is in the image, and every drive comes with `fetch`.

| `--detector-model` | Network and input | Parameters | Labelled road users found |
|---|---|---|---|
| `yolox_nano` | YOLOX-Nano, 640x192 | 0.9 M | 91.5 % |
| `yolox_s` | YOLOX-S, 640x192 | 9 M | not scored |
| `yolox_s-1024` | YOLOX-S, 1024x320 | 9 M | not scored |
| `yolox_m-1024` | YOLOX-M, 1024x320 | 25 M | 97.0 % |
| `yolox_l-1248` | YOLOX-L, 1248x384, the frame's own resolution | 54 M | 97.4 % |
| `yolox_x-1248` | YOLOX-X, 1248x384 | 99 M | 97.5 % |

The last column is `score` over the four drives `fetch` downloads to start with: the
share of the 2,351 road users KITTI's labellers marked that the model found. I use
YOLOX-L on an i7-12700 with an RTX 3060; try the sizes and keep the one that works
best on yours. The two biggest are 16-bit files made for the GPU
(`--detector-backend cuda`), and on a CPU the bigger sizes drop frames.

The lidar model is PointPillars, trained on KITTI's own lidar to find cars,
pedestrians and cyclists as 3D boxes. Against the same labels it finds 90.8 % of the
vehicles ahead over those drives, though from one drive to the next it ranges from
under half to nearly all. The full tables are in
[docs/measurements.md](docs/measurements.md#the-biggest-models).

Any of the 38 drives KITTI labelled can be replayed and scored: `fetch --list` names
them, `fetch <drive>` downloads one, `run --drive <drive>` replays it, and
`score runs/<name> <drive>` grades the run.

## Output

The terminal prints a few hundred lines; these are the ones to read first:

```text
fusion sets: COMPLETED 154 (...) | DEGRADED 0 (...) | EXPIRED 0 = dropped 0 + late 0 + absent 0 | NEVER REACHED track 0 of 154 sweeps
ANSWER most urgent: sweep 130, 2.24 s to contact -- object 2329 (car 0.95, lidar car 0.98) at 10.04 m, ...
INVARIANT edge=cam0->camdet delivered=154 dropped=0 admitted=154 -> OK
```

`fusion sets` accounts for every sweep: paired with its own frame, paired with a stale
one, expired (split by reason), or never reached the fusion. The exit code (`echo
$?`, or `$LASTEXITCODE` in PowerShell) is 0 when every invariant held, 2 when one
failed or a flag was mistyped, 3 when a stage panicked, and 1 for any other error.
Each run also writes a folder, `runs/<name>/`:

| File | What it holds |
|---|---|
| `evidence.csv` | One row per sample per hop, and one per drawing sent to the viewer. Key columns: `edge`, `stage` and `seq` (the source frame's number, shared by every hop, so rows join on it); `due_ns`, `arrival_ns`, `queue_wait_ns` and `measurement_age_ns`; `outcome` and `reason`; `storage_id`, `bytes_alloc` and `payload_bytes`. |
| `summary.json` | The terminal's numbers as JSON, for comparing runs: counts and latency percentiles per stage, each detector, the fusion and the answer. |
| `run.json` | How the run was set up: arguments, git commit, drive, each detector's model, hash and runtime, and the clock's origin. |
| `events.csv` | Run-level events, such as a gap in the source and the shutdown. |
| `cam_det.arrows`, `cam1_det.arrows` | Every detection of the left and of the right camera: one Arrow batch per frame, with box, class and confidence. |
| `pillar_det.arrows` | Every box the lidar detector reported: one batch per sweep, each box nine numbers (centre x, y, z, length, width, height, heading, class, score). A box behind the car has a negative x. |
| `fused.arrows` | Every fusion result: one batch per sweep, with its tracks and the camera frames used. |
| `cam0.rrd` | The recording, only with `--rerun rrd`. |
| `_COMPLETE` | Written last. Without it, the run did not finish and the other files may be cut short. |

The `.arrows` files are Arrow IPC streams, which any Arrow library reads, and each
exists only when its stage runs. A `--name` used before is written over, but files of
stages the new run does not run stay from the old one, so use a new name when the
flags change.

## Tests

```sh
docker compose run --rm test      # all 458
docker compose run --rm test --   # the 443 that need no drives
```

The tests run in Docker on the code in this checkout, so an edit needs no rebuild of
the image; the first run compiles the project. Most are unit tests of each stage,
about a hundred run `pipes` itself on small synthetic drives, and fifteen replay two
of the drives `fetch` downloads to start with, with the real models.

Before a change is called done it is also watched: a run in Docker streaming to the
open viewer window, as in [First run](#first-run), not a headless one.

## How it works

1. **Drivers and one admission.** A driver per sensor, each on its own thread, reads
   the drive from disk and releases each sample when the real sensor would have
   delivered it. All of them hand their samples to a single admission, which puts
   every sample into one arrival order and onto the queue of each stage that reads
   it. The stages admit their results the same way.
2. **Bounded queues.** Every connection between stages is a queue with a fixed
   capacity. A full data queue drops its oldest sample, so a push never waits and a
   slow stage cannot hold up a sensor, and each drop is recorded with its reason. The
   viewer is a consumer like the others: one that freezes loses drawings and nothing
   else.
3. **Written once, read in place.** Each driver loads its sample into an Arrow buffer
   owned by Pipes, and every later stage reads its input straight out of its
   producer's buffer. Every evidence row records the address of the buffer read and
   the bytes the stage allocated, and the run checks that a stage read the very
   buffer its producer wrote.
4. **The lidar chain** has four stages. `reduce` averages the sweep's roughly 120,000
   points into 20 cm cubes. `detect` removes the ground and groups what is left within
   30 m into clusters, each reported as a box. `track` follows the boxes from sweep to
   sweep and gives each an id, a velocity and an age. `state` turns the tracks into
   the answer. Per sweep, 1.95 MB of points becomes a 15 kB answer.
5. **The lidar detector** gives those tracks names. The chain finds structure, not
   objects: walls, hedges, cars and fragments of each. PointPillars reads the same raw
   sweep in place in a stage of its own, `pillars`, once for the 69 m ahead and, on
   the GPU, once more for the 69 m behind. A track whose centre lies inside one of its
   boxes scoring 0.3 or more takes that box's class: car, pedestrian or cyclist.
6. **The camera detectors** name what the cameras see. A YOLOX network runs on each
   camera's frames in a stage of its own (`camdet`, `camdet-right`), reading the frame
   in place. Every model is frozen: its SHA-256 is checked at load and nothing learns.
7. **The car's own motion** comes from the GPS/IMU. The tracker carries every track
   through what the car did since the last sweep, so a parked car stays where it is
   and a track's velocity is the object's own. The car's path is the arc its speed and
   turn rate describe, and "in the car's path" means within half the car's width of
   it.
8. **Fusion** pairs the sensors by time. A sweep is a 103 ms rotation and a camera
   frame an instant, so each sweep pairs with the frame whose instant falls inside it,
   and waits up to one sweep for a detector that is late. A track's 3D box is
   projected into the image, and the track takes a detection's class when the two
   boxes overlap by more than half, and keeps it only when the right camera confirms
   it. A sweep whose left frame was lost is answered from the right frame. A camera or
   a lidar detector that sends nothing for three sweeps is declared silent: the fusion
   goes on without it and says so.
9. **Time to contact.** For every track the answer gives the distance along the path
   and how fast that gap is closing, and for one in the car's path the time to
   contact: how long until the car reaches it if nothing changes. It counts only the
   closing speed the track's age has established, because a young track's speed is
   mostly noise, so a newly seen object gets one a second or two after it is first
   tracked.
10. **Evidence.** Every sample leaves one row per hop in `evidence.csv`. When the
    drive ends, the run checks conservation rules against those rows: on every
    connection delivered plus dropped equals admitted, every sensor frame is admitted
    or recorded as missing, every answer holds every track. Each is printed as an
    `INVARIANT` line, 51 of them on the GPU run, and the run exits 2 if one fails. A frame missing from the source is recorded as missing, and
    a late camera frame is never used without saying so.

## What is slow

On an RTX 3060 with the viewer window open, the answer comes about 43 ms after each
sweep ends, and the lidar detector is why. A sweep is one 103 ms rotation, and the
detector cannot start until the rotation is complete:

| Part | Starts | Done, after the sweep ends |
|---|---|---|
| Lidar chain (`reduce`, `detect`) | when the sweep ends | about 11 ms |
| Camera detectors, YOLOX-L | when the cameras fire, about 41 ms before the sweep ends | about 13 ms |
| Lidar detector, ahead and behind | when the sweep ends | about 42 ms, 34 of them the network |

The cameras cost more per frame, but they work while the lidar is still turning. The
lidar detector cannot, so the fusion waits about 24 ms a sweep for its boxes. On a CPU
it takes about 240 ms a sweep, longer than the sweep itself, which is why it needs a
GPU. Today `--lidar-views ahead` drops the pass behind the car, and the answer comes
about 25 ms after the sweep.

What could make it faster, none of it tried yet:

- Send the answer after the pass ahead and run the pass behind after it: nothing
  behind the car is in its path, so the answer should not wait for it.
- Start the pass ahead as soon as the lidar has swept the front of the car, before
  the rotation ends.
- Run the network on TensorRT, which NVIDIA built this PointPillars for, instead of
  ONNX Runtime.
- Give the detectors a GPU the viewer window does not also draw on.

## Limits

- It only replays recorded data: no live sensor drivers, no clock synchronization
  between real sensors, and no ROS 2 output yet.
- The time to contact rests on what the tracker measures, so a newly seen object
  gets one a second or two after it is first tracked.
- The corridor is the arc of one instant, a guess past about ten metres of a bend.
- The lidar's sweep is not corrected for the car moving during its 103 ms rotation.
- The lidar detector needs a GPU to keep up with the sweeps and was trained on KITTI,
  whose licence is non-commercial.
- The right camera's picture is not on the dashboard, and a camera-only object is not
  in the answer, because it has no velocity.
- The distance is to the near face of a track's box, so a wide object that only
  partly overlaps the car's path can read too near.

## Next

The next step is to send the answer out as a ROS 2 object list: one message a sweep
with every tracked object's position, size, heading, velocity and class, so a ROS node
gets what the dashboard shows. It is not built yet.

## Design

The architecture is Ethan Boos's, and his
[architecture and literature review](docs/architecture-and-literature-review.md)
explains the reasoning behind it and the prior work it builds on.
[docs/measurements.md](docs/measurements.md) has each experiment with its tables, and
[docs/sprint-2-plan.md](docs/sprint-2-plan.md) what was built, in what order and why.
The code is MIT-licensed (see `LICENSE`); Ethan's design document is his own and not
covered by that licence.
