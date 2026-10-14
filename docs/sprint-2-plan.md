# Sprint 2 plan: GPS/IMU, the second camera, and a faster camera

_Written October 2026, after the dashboard rewrite (commit 90edd6d)._

## What sprint 1 left open

- Speeds are relative to the car, and the tracker cannot subtract the car's
  turning: in a turn every stationary object sweeps sideways in the lidar
  frame, so tracks split, velocities are wrong and the closing speed is only
  right on a straight road.
- Only the left colour camera (`image_02`) is used. The right one
  (`image_03`) is on disk with the same timestamps and its own calibration.
- The detector takes 36 ms a frame on one CPU thread. The camera's result is
  ready about 4.5 ms before the sweep ends; 13 ms more and the fusion starts
  waiting. There is no room for a second camera on the same thread, or for a
  bigger model.
- About 94 % of the lidar tracks in the camera's view never get a class:
  foliage, walls and fragments of larger objects are tracked but not named,
  and some of them end up as the answer.

## Facts about this machine

- CPU: i7-12700, 8 performance cores plus 4 efficient cores, 20 threads.
- GPU: RTX 3060, 12 GB, CUDA 13.3, reachable from WSL (`/dev/dxg`) and from
  Docker (`--gpus all` works; the `nvidia` runtime is installed).
- WSL has 7.6 GB of RAM and 2 GB of swap. Release builds have been killed
  for memory before (exit 137). A `.wslconfig` with `memory=16GB` would help
  if the host has it.

## Order, and why

1. **OXTS (GPS/IMU) as a third stream, and ego-motion compensation.**
   Independent of everything else, and the most certain win: it fixes the
   turning problem and makes the closing speed mean what the headline says.
2. **The right camera as a fourth stream, with its own detector thread, and
   two-camera agreement in the fusion.** Needs the driver and calibration to
   take a camera index, and the fusion to pair two camera streams. A second
   detector thread costs no time on this CPU (12 cores), so this does not
   wait for step 3.
3. **A faster camera stage, tuned to this machine.** Measure before choosing:
   tract on one thread (today), tract on several, ONNX Runtime on the CPU,
   ONNX Runtime on the GPU. Pick the one that keeps runs repeatable where it
   can, and record the choice in `run.json`.
4. **The unnamed tracks.** Measure what they are first, then decide between a
   model that knows more classes (a segmentation model trained on street
   scenes names vegetation, buildings and poles; COCO does not) and lidar-side
   rules. Depends on step 3 for the time budget.

ROS 2 output, MCAP recording and live drivers stay deferred.

## Step 1: OXTS

**Driver.** `pipes-kitti/src/oxts.rs`: `OxtsDriver`, built like `Cam0Driver`
on `FrameIndex` over `oxts/data/*.txt` and `oxts/timestamps.txt`. One Arrow
batch per sample, one row, the 30 fields as columns (23 `f64`, 5 `u32` status
fields, plus the frame). Paced to its own timestamp, `Tov::Time`. Stream
`StreamId::OXTS` (already declared, id 2), edge name `oxts`. A frame the
source lacks is `Missing { absent_in_source }`, as for the other sensors.

**Edge.** `oxts->track`, cap 64, drop-oldest, drained by the `track` thread
the way `cam_det->track` is. The sample paired with a sweep is the one of the
sweep's frame number (KITTI's `_sync` OXTS is one row per frame, stamped
within a few ms of the camera trigger); the frame before stands in when it is
missing, flagged `ego_stale`; none means `ego_none` and no compensation.

**Tracker.** Before association, carry every live track (`pos`, `lo`, `hi`,
`first_pos`) through `EgoMotion::carry` over `dt` with the sweep's `vf`, `vl`
and yaw rate (`wu`, the vehicle-frame rate). Velocity then becomes velocity
over ground. The gate's yaw term goes once compensation is on.

**State.** Closing speed = the car's forward speed minus the object's ground
velocity along the line of sight (relative velocity, as before, but now
right in a turn). The answer's text names the ego speed. The corridor stays
straight for now; a curved one (radius `vf / yaw_rate`) is a follow-up.

**Evidence.** Driver rows on `oxts`, delivered rows on `oxts->track`, and the
track row's `reason` carrying `ego_stale` / `ego_none`. `summary.json` gets
an `oxts` section (admitted, missing, speed range, yaw-rate range) and the
track section gets how many sweeps were compensated. Invariants:
`driver=oxts admitted+missing=n_frames` and the edge's conservation.

**Check.** Drive 0005 has two turns (yaw rate up to 0.25 rad/s). Compare the
healthy run before and after: fewer track resets and births during the
turns, lateral velocities of stationary objects near zero, the same answer
on the straight sections.

**Result (October 2026).** Done; the measurements are in `measurements.md` under
"Without the car's own motion", from runs streaming to a live Rerun viewer
(a headless run gives identical figures: nothing was dropped). With the
viewer on, the detector takes 42 ms a frame (p95 49) against 37 headless,
and the camera's result comes 1 ms after the sweep ends instead of 4 ms
before it, so the fusion waited on 6 of 154 sweeps, for 0.1 ms a sweep. Of the tracks older than 0.5 s, 74 % move
under 1 m/s over ground with the stream against 7 % without; lateral speed
median 0.18 m/s against 1.47; the turning sweeps look like the straight ones
(median speed 0.32 m/s against 0.37). Ambiguous associations fall from 91 %
to 62 %. The narrower gate starts 40 % more tracks, which is the fragment
problem of step 4, not a tracker fault. The most urgent answer keeps its
object and corrects its closing speed from 4.2 to 5.8 m/s (the car's own
speed was 6.05 m/s). The corridor stays straight; a curved one (radius
`vf / yaw_rate`) is still a follow-up.

## Step 2: the right camera

- `Cam0Driver` becomes `CamDriver::open(root, date, drive, cam)` with the
  camera index; `image_02` is `cam0`, `image_03` is `cam1`. New stream ids
  `CAM1` (8) and `CAM1_DET` (9); `Calib::load(root, date, cam)` parses
  `P_rect_03` and `S_rect_03` (`R_rect_00` is the one the devkit uses for
  every camera).
- A second `camdet` thread on the right camera's queue, sharing the loaded
  model (the plan is immutable; each run makes its own state).
- `track_thread` drains both camera queues and pairs each with the sweep by
  the same rule. Each track is projected into both images and matched in
  each. Agreement rule: a track takes a class only when both cameras match
  it with the same class; one camera alone is recorded as `one_camera` and
  shown in the teal lidar-only colour with its class in the label.
- Camera-only detections are kept only when both cameras see them: same
  class, same image row within a few pixels, same height within 20 %, and
  the left box right of the right box by 0 to about 80 px (rectified stereo,
  54 cm baseline, so the shift is `721 × 0.54 / Z`). The shift gives a
  range, `Z = 389 / shift`, which becomes the camera-only object's distance.
- The dashboard's camera view shows the left image; a second tab shows the
  right one with its boxes. The counts plot gains `one_camera`.

**Check.** Matches per run on drive 0005 (482 today, left camera alone):
how many survive agreement, how many camera-only boxes gain a range, and
whether the answer changes on the straight sections (it should not).

**What the second camera is for.** Four things, each measured by the run:
agreement (a class is kept only when both cameras match the track with the
same class; one camera's match is `one_camera`, which the answer does not
repeat), a stereo range for camera-only detections (kept only with a partner
in the other image), a second range on every confirmed track to check the
lidar's distance against, and redundancy (a sweep whose left frame was
dropped or late is answered from the right frame instead of expiring, with
every match one camera's and the set saying so). With the right camera on
and its frame missing for a sweep, the left's matches are demoted to
`one_camera`: a two-camera run never quietly becomes a one-camera one.

**If one stream breaks.** Each camera has its own queues, so a stalled
detector drops only its own frames. A camera from which nothing has
arrived for three sweeps running, none of them pairing, is declared
silent at the fusion: it is not waited for, the other camera's set is
built on one camera's terms (classes trusted, as with one camera), the
set and the evidence say `pair_silent`, the summary counts the sweeps each
camera was out for, and the picture's corner says which camera is silent;
a frame of it arriving or pairing brings it back. Arrivals count because
a camera late by less than the wait is alive: counted by pairings alone,
the GPU detector's slow first frame silenced both cameras for a whole run.
The hidden `--right-delay-ms` hook breaks the right stream on purpose, as
`--consumer-delay-ms` does the left, and the gated test runs both.

**Done (October 2026).** Wired end to end: `--right-camera auto|on|off`,
the `cam1` driver thread, the `camdet-right` stage on its own thread with
the shared model, the `cam1->camdet` and `cam1_det->track` edges, the
fusion's second buffer with one shared deadline for both cameras, the
`cam_side`/`cam1_*` columns on the track and answer batches, the provenance
join of the right frame (`pair_check`), `summary.json`'s `cam1`,
`camera_detector_right`, the fusion's two-camera tallies and the stereo
check, the invariants `driver=cam1`, `stage=camdet-right` and the right
pairing's, the dashboard's one-camera population (teal, `car? 0.62`), the
stereo range on camera-only labels, and the right camera's node on the
pipeline graph. The right picture itself is not on the dashboard. The
measurements are in `measurements.md` under "Two cameras", from runs streaming to
a live viewer. Against the check above: of the 488 matches on drive 0005,
393 survive agreement, 95 do not and 54 are the right camera's alone; 363
of 739 camera-only boxes gain a range and 376 are dropped; the stereo range
of a confirmed track lands on the lidar box's centre (median ratio 1.00,
1.03 on drive 0009); the answer on the straight sections does not change
(the most urgent object is the same car on sweep 140); and with the left
detector slowed, the 40 sweeps whose left frame was lost are answered from
the right frame where one camera alone expired 38. The second detector
thread costs no time: both stages take the same per-frame time as one
alone, measured back to back.

## Step 3: camera speed and machine tuning

- A benchmark binary, `pipes bench-detector`, that runs the detector over N
  frames of a drive and prints preprocess, inference and total per frame, on
  each backend compiled in.
- Backends to measure: tract one thread (today); tract with
  `multithread-mm`; ONNX Runtime CPU with 1, 4 and 8 threads; ONNX Runtime
  CUDA. Expect roughly 36 / 15 / 10 / 3 ms. Record whether two runs give the
  same bytes on each.
- Choose by: time per frame, repeatability, and what it costs the Docker
  image (CUDA + cuDNN runtime adds about 1.5 GB and needs `--gpus all` in
  compose). The default stays repeatable; the GPU is a flag,
  `--detector-backend`.
- CPU: pin the two detector threads and the lidar chain to performance
  cores (`sched_setaffinity` through `libc`, Linux only, off by default);
  measure whether it cuts the camera's tail. Reuse the preprocess buffers,
  which allocate 5.6 MB a frame today.

**Check.** The benchmark table in `measurements.md`, and the zoom plot on the Demo
page showing the camera diamond further below the sweep's end.

**Done (October 2026).** `pipes bench-detector --model M --backend B
[--threads N]` times one model on one runtime over a drive's first frames
and says whether two passes give the same bytes. Three backends behind
`--detector-backend`: `tract` (the default, one thread, repeatable
anywhere), `ort` (ONNX Runtime 1.22.0 on the CPU, `--detector-threads`,
hidden, four by default) and `cuda` (ONNX Runtime's CUDA provider, cuDNN's
algorithm chosen by rule and TF32 off, so it repeats on one machine). ONNX
Runtime is loaded at run time from `ORT_DYLIB_PATH` (`ort` crate, pinned,
`load-dynamic`): a machine without it still builds and runs the default.
Four models behind `--detector-model`: YOLOX-Nano at 640x192 (the
default), YOLOX-S at 640x192 and at 1024x320, YOLOX-M at 1024x320, each a
file derived from Megvii's published ONNX by `scripts/prepare_models.py`
(the declared input set to the pipeline's size, which ONNX Runtime insists
on) with its own pinned sha256. The Docker `model` stage derives them,
the `test` and `runtime` images carry the CPU half of ONNX Runtime, and
the new `gpu` image (`docker compose run --rm pipes-gpu`, NVIDIA's CUDA
12.6 + cuDNN 9 runtime) carries the CUDA provider. `run.json` and
`summary.json` record the model, its hash, its input and the backend. CPU
core pinning was not done: the measurements did not need it.

**Result.** Alone, ONNX Runtime runs Nano in 15 ms on one CPU thread
against tract's 33 (the same detections), 9.5 ms on four; the GPU runs
Nano in 7 ms, S at 1024x320 in 23 (14 with TF32) and M at 1024x320 in 52
(21 with TF32), every configuration the same bytes twice. In the pipeline
the two cameras' sessions share the GPU and each takes about twice its
solo time, so the configuration whose result is ready when Nano's is today
is M on the GPU with TF32 (46 ms a stage, camera done +5 ms after the
sweep's end, 2,003 detections and 476 confirmed tracks against Nano's
1,227 and 393); the GPU image runs that by default. ONNX Runtime on the
CPU is the cheap win: Nano ready 25 ms before the sweep ends. Two things
the GPU runs found: cuDNN's default algorithm runs Nano's depthwise
convolutions in 168 ms (the heuristic choice fixes it, and is what a run
uses), and the fusion's silence rule had to count arrivals, not pairings
(the GPU's slow first frame, now paid at load, silenced both cameras for a
run). Next on the GPU: one batched session for both cameras instead of two
taking turns. Numbers: `measurements.md`, "The detector on the GPU, and a bigger
model".

## Step 4: the unnamed tracks

- Measure first, offline, from `fused.arrows` and `evidence.csv` of a
  healthy run: of the lidar-only tracks in frame, how many are off the road
  (lateral offset over 4 m), how tall, how dense, how long-lived, and how
  many become the answer.
- Prototype a street-scene segmentation model (Cityscapes classes:
  vegetation, building, pole, fence, ...) in Python on the drive's frames,
  project each lidar-only track's box into it, and tally the dominant class.
  If vegetation and buildings account for most of them, add the model as a
  second camera stage on the chosen backend and give every lidar track the
  label under its box. If not, add lidar-side rules instead.
- Either way, the answer only considers tracks whose class is a road user
  or unknown-and-on-the-road.

**Check.** The share of lidar-only tracks with a class on drive 0005, and
no answer that is a bush.

**Done (October 2026), by another route.** The plan above was a camera
model that says what the lidar's clusters are. What was built is a lidar
model that finds the road users directly, which is the half of the question
the answer needs: PointPillars, trained on KITTI's own lidar for cars,
pedestrians and cyclists, as a frozen detector in a stage of its own.
`--lidar-detector on` (off by default: the network is a GPU's job), a
`pillars` stage on its own thread reading the raw sweep in place beside
`reduce`, the `velo->pillars` and `pillar_det->track` edges, the stream
`pillar_det`, and `pipes bench-lidar`. The model is NVIDIA's published
export of OpenPCDet's PointPillars; `scripts/prepare_models.py` replaces its
one TensorRT-only node with standard ONNX and pins the result's hash, and
the Rust decode was checked box for box against a numpy reference. The
fusion waits for a sweep's boxes inside the cameras' deadline and gives a
track seen in that sweep the class of the box its centre lies in, from a
score of 0.3: at the network's own 0.1 it calls lamp posts pedestrians. A
detector that stops answering is declared silent after three sweeps, like
a camera. New columns on the track and answer batches, `summary.json`'s
`lidar_detector` and `naming` blocks, six more invariants, the violet boxes
in the lidar view and a `pillars` node on the pipeline graph.

Against the check: on drive 0005 the lidar detector names 1,407 of 23,821
track observations, 1,055 of them tracks no camera named, and where a
camera named the same track the two agree on all 352 (on 400 of 402 with
YOLOX-M). The other half of the check, no answer that is a bush, took a
second change: the corridor now follows the car's turning (an arc of
curvature yaw rate over speed), and an unnamed track is in it by its centre
rather than by its box. The track that decides is then one a sensor named
on 88 of the 131 sweeps that have one on the path, against 69 of 125
without the lidar detector. Numbers: `measurements.md`, "The lidar detector".

**The GPU, measured.** The cameras were going to stay on the CPU so as not
to compete with the lidar detector. They do not compete on this host: each
network runs faster the more the GPU has to do (PointPillars 36 ms a sweep
alone on the GPU, 18 beside YOLOX-M, which takes 33 ms a frame against 47
alone), most likely because a GPU that idles between sweeps slows its
clocks. With everything on the GPU the answer comes 21 ms after the sweep's
end; with the cameras on the CPU, 38. The GPU image runs all three sessions
on the GPU by default, and the batched camera session planned at the end of
step 3 is not needed: nothing waits for the cameras.

## Step 5: the decision

Asked for on 13 October 2026: a final decision on what the car should be
doing at each second, from the GPS/IMU, the cameras and the lidar, fused.

- `pipes-kitti/src/path.rs`: the path as an arc, a track's place on it
  (offset, distance along, whether it is on the corridor), the gap to its
  box along the path and its closing speed along the path's direction.
- `pipes-kitti/src/decide.rs`: the rule. For every track on the path, the
  deceleration that stops the gap closing 5 m short; the most demanding
  decides, `continue` under 1 m/s², `slow` from 1, `brake` from 3; both
  thresholds halved for a pedestrian or cyclist, for a class the sensors
  disagree on, for one a single camera gave, for one a stale frame gave;
  raised at once, lowered after three sweeps that ask for less. The four
  numbers are stated, not derived, and each decision records what it was
  made on.
- The rule acts on the closing speed a track's age has established: the
  measured one less two tracker gates over the track's age. This is the
  part that was found by checking, not planned (below).
- The `state` stage decides inside its measured window, allocating
  nothing; the answer batch carries the decision in eleven columns; the
  `state-sink` consumer reads it back, re-derives it from its own numbers,
  checks it against the answer's records, and keeps one row a sweep
  (`decisions.csv`). The terminal prints the most demanding decision of
  each second, `summary.json` has the same, and one more invariant.
- The dashboard's headline leads with the decision and the lidar view
  draws the corridor in its colour. No new view or plot.
- The answer's time to contact uses the same established closing speed, so
  the headline cannot say `CONTINUE` beside a time to contact made of one
  sweep's noise.

**Check, and what it found.** A recorded drive says what happened next, in
the GPS/IMU rows after each sweep, which the pipeline never reads.
`scripts/check_decisions.py` uses them as a witness: whether the car went
on to drive over the place of the track that decided, and what the driver
did in that second. The first version of the rule, on the tracks'
velocities as measured, asked for braking on 4 sweeps of drive 0005 and 29
of drive 0013, where no driver brakes: every one a track a few sweeps old,
and the 29 a single vehicle ahead that the tracker lost and found on every
sweep because it was pulling away faster than the tracker's 50 km/h bound.
With the established closing speed the rule asks for braking on no sweep of
the four drives, and for slowing on 25, 9, 0 and 0: 15 for a cyclist the
car follows round a bend (on the driven path), 15 for cars parked on the
inside of a bend the car was leaving (beside it by more than 2 m), 4 where
the drive ended first. Those 15 are the arc's limit, which the same script
measures: on drive 0005 the car is within half its width of the arc after
10 m on 79 % of frames and after 20 m on 28 %. Numbers: `measurements.md`,
"The decision".

**Open after this step.**

- The path. One instant's arc is a guess past ten metres of a bend. A
  clothoid fitted to the last half second was tried and is no better. What
  would help is knowing the road: lane markings from the cameras, or the
  lidar detector's parked cars as the road's edge.
- The tracker's speed bound. A vehicle faster than 50 km/h over ground is
  lost and found again on every sweep, so it never has an established
  velocity and the decision cannot act on it before 5 m. The gate was built
  for drive 0005; drive 0013 is faster.
- A time to contact for young tracks. The lidar detector's boxes are a
  steadier position than a cluster's centre, and following them from sweep
  to sweep would give a named object a velocity sooner.
- Nothing on the four drives needs braking, so `brake` has unit tests and
  no recorded scene. A drive with a hard stop would be the next data to
  fetch.

## Step 6: the biggest models each sensor can carry

Asked for on 13 October 2026: the biggest and best models for every sensor,
pushed to the limit if bigger is better, and what the lidar model says on
the display, with the dashboard otherwise as it is.

- A witness first. "Better" had only been agreement between the sensors.
  KITTI publishes the boxes its labellers drew on these four drives
  (tracklets), which the pipeline never reads. `fetch` now downloads them
  and `docker compose run --rm score runs/<name> <drive>`
  (`scripts/score_detectors.py`) counts how many labelled road users each
  detector found, from the run's own `cam_det.arrows` and
  `pillar_det.arrows`.
- Cameras. YOLOX-L and YOLOX-X joined the catalog at 1248x384, the frame's
  own resolution, as files of 16-bit weights for the GPU
  (`--detector-model yolox_l-1248`, `yolox_x-1248`). Half precision is what
  made them fit: about one and a half times faster on this GPU, the same
  objects found. YOLOX-L is the GPU image's default. YOLOX-X is not: it
  finds no more vehicles, and with it the answer comes 62 ms after the
  sweep's end instead of 35.
- Lidar. No bigger lidar network was found that ONNX Runtime can run: the
  stronger KITTI detectors need sparse 3D convolutions. PointPillars was
  pushed two other ways with the same frozen weights. A 16-bit file for the
  GPU, 14 ms a pass instead of 18. And a second pass a sweep over the sweep
  turned half round (`--lidar-views`, `around` on the GPU), so it names what
  is behind the car as well; one NMS over both passes.
- Behind the car nothing is labelled, so the second pass was scored by
  carrying: a parked car's label, moved through the car's own GPS/IMU
  motion into the sweeps after the car has passed it. 93 % of those within
  15 m are found, 73 % between 15 and 30 m, 25 % beyond.
- Dashboard. No new view. The lidar views draw the detector's boxes scoring
  0.3 or more only, the picture's corner stamp says what the lidar named on
  the sweep, and the zoomed timing plot reaches 46 ms so the answer stays on
  it. The corridor lines added in step 5 stay: they were hidden until
  Jonathan had seen them, and he asked for them back.
- Found on the way: every camera batch's metadata named YOLOX-Nano whatever
  model had run. Fixed, with a test.
- Every size stays in the catalog, YOLOX-X included: which one fits depends
  on the machine, and the README's First run has a table of what each
  service runs.
- The timings here were taken streaming to a viewer with no window. With the
  viewer window open on the same GPU, which is how a run is watched, the
  answer comes 41 to 46 ms after the sweep's end with YOLOX-L and 80 or more
  with YOLOX-X, whose lidar boxes then miss the deadline on some sweeps.
- The README had more than doubled over the sprint. It was cut back to the
  length it had before, and then rewritten from the top to be picked up fast:
  a Docker-only setup that brings every model and every drive, a Commands
  section built from what `--help` lists, the GPU models and the lidar
  detector first, and no Experiments section. Every experiment's write-up and
  table is in `docs/measurements.md`.
- Then the decision layer of step 5 came out, at Jonathan's request: on these
  drives it said `continue` almost everywhere, and most of its few `slow`s
  were for parked cars beside a bend. The answer keeps the time to contact;
  `decisions.csv`, its invariant and the `check` service went with it.

**Check.** Against the labels, four drives, viewer on: YOLOX-L finds 97 % of
the 2,351 labelled road users in view where YOLOX-Nano finds 92 % (98 % of
the vehicles against 93 %, 76 % of the pedestrians against 45 %), and the
lidar detector 91 % of the 2,428 ahead. The cost of looking behind is 17 ms
on the answer, and nothing behind the car can be in its path; `--lidar-views
ahead` gives it back. Numbers: `measurements.md`, "The biggest models".

**Open after this step.**

- The lidar detector misses half the labelled vehicles on drive 0005 beyond
  15 m, and finds 98 % or more on drives 0009 and 0013. Why that street is
  hard for it has not been looked at.
- A name behind the car has no second sensor to confirm it, and the answer
  uses it only as a label: nothing behind the car is in its path or has a
  time to contact. Watching what is behind (a vehicle closing from the rear)
  would be the first use.
- A bigger lidar model means a runtime with sparse convolutions, or a
  detector trained on another lidar (Autoware's CenterPoint, nuScenes).
