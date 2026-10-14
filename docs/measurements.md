# Measurements

What the README's short claims rest on: each experiment as it was written up when it
was made, with its commands and tables. Each command is for a second terminal, with
the Rerun viewer open in the first, as in the README's First run. Unless a section names another, the drive is
0005 of 2011_09_26 and the machine an i7-12700 with an RTX 3060, Docker under WSL 2.

The timings are this machine's, and were taken streaming to a Rerun viewer with no
window. With the viewer window open on the same GPU the answer comes later: with
YOLOX-L 41 to 46 ms after the sweep's end instead of 35, and with YOLOX-X 80 ms or
more instead of 62.

## The healthy case

```sh
docker compose run --rm pipes run
```

Both detectors see all 154 frames, every sweep pairs with its own frame from each
camera, the fusion confirms 393 tracks with both cameras (488 matches with the left
camera alone, `--right-camera off`), and all 45 invariants hold.

## A slow camera

```sh
docker compose run --rm pipes run --consumer-delay-ms 100 --cap 1 --pair-wait-ms 300 --right-camera off
```

The detector sleeps an extra 100 ms per frame, so its one-frame queue drops frames
(38 of 154 on one run; it varies). The fusion expires exactly those sweeps, as
`pair_dropped`, and every invariant holds, because the drops were declared. With the
right camera on, those sweeps are answered from its frame instead; see
[Two cameras](#two-cameras).

## A stale camera

```sh
docker compose run --rm pipes run --consumer-delay-ms 50 --pair-wait-ms 0 --pair-stale-ms 500 --right-camera off
```

The detector sleeps an extra 50 ms per frame, so each frame is ready only after its
sweep has been processed, and the fusion, which no longer waits, takes the previous
frame, 93 ms old. Matches fall from 488 to 416, 153 of the 154 answers are built on a
stale frame (the first sweep has no previous frame and expires), and every one of
them says `STALE camera 93 ms`. With the right camera on, the fusion would take the
right frame, which is on time, instead of the stale left one.

## Why the fusion waits

```sh
docker compose run --rm pipes run --consumer-delay-ms 50 --pair-wait-ms 0 --right-camera off
```

The detector sleeps an extra 50 ms per frame. No frame is lost, but each arrives
after its sweep has been processed, so all 154 sweeps expire as `pair_late` and
there is no answer.

## A gap in the source

```sh
docker compose run --rm pipes run --drive 2011_09_26_drive_0009_sync
```

Drive 0009 has no lidar for frames 177–180 of 447. The run records the four as
absent, answers the other 443 sweeps and exits 0. It takes 46 seconds.

## Any other drive

```sh
docker compose run --rm pipes drives
docker compose run --rm pipes run --drive 2011_09_26_drive_0013_sync
```

The first command lists the drives; the second replays one of them. Drives 0005,
0009, 0013 and 0048 of 2011_09_26 have camera and lidar. A drive without
`velodyne_points`, such as a camera-only copy of 2011_09_28_drive_0001, replays the
camera alone.

## Without the detector

```sh
docker compose run --rm pipes run --detector off
```

The chain without the camera model: tracks still get image boxes but no class, and
nothing is camera-only.

## Without the car's own motion

```sh
docker compose run --rm pipes run --oxts off
```

The tracker falls back to velocities relative to the car and a gate wide enough to
absorb the car's turning. Drive 0005 is turning on 132 of its 154 frames. With the
stream, 74% of the tracks older than half a second move slower than 1 m/s over ground,
which is what a street of walls, parked cars and bushes should look like; without it,
93% appear to move faster than 1 m/s, at the car's own speed, and the share of
detections with more than one track inside the gate rises from 62% to 91%. The
narrower gate also starts more tracks (3,442 against 2,449), because a bush that
breaks into different fragments from sweep to sweep no longer re-matches across 3 m.
Without the stream the car's path is also taken as straight ahead: something is in
the straight corridor on 138 of the 154 sweeps against 125 on the arc the car is
following, much of it walls and parked cars the car is turning past. These figures are from runs streaming to a live viewer, and a
headless run gives the same ones to the last digit: with no sweep or frame lost, the
tracker's output does not depend on the load.

## Two cameras

```sh
docker compose run --rm pipes run
docker compose run --rm pipes run --right-camera off
```

The first command is the default run, which now reads both colour cameras; the second
is the same drive with the left camera alone. With one camera the fusion matched 488
tracks to detections over drive 0005. With two, 393 of those were confirmed by the
right camera (a detection of the same class covering the track's box in the right
image), 95 were not and became `one_camera`, and the right camera matched 54 tracks
the left had missed. Of the 739 detections no track matched, 363 had a partner in the
right image and got a range; the other 376 were dropped. The range of every confirmed
track was checked against the lidar: the stereo range lands on the lidar box's
centre (median ratio 1.00 on drive 0005 and 1.03 on drive 0009, within 1 m or 10 % on
52 % of the 387 tracks on 0005) and reads 11 % farther than the box's near face, as
the disparity of a silhouette should. The answer is the same as with one camera: the
most urgent object is the same car on sweep 132, 12.7 m away and closing at
5.5 m/s. On drive 0009, 1,092 of 1,218 matches were confirmed and 1,361 of 2,138
camera-only boxes got a range. The second detector stage costs no time: measured back
to back with the viewer on, each stage takes 44 ms a frame at the median with two
cameras against 43 to 44 ms with one, because the two run on separate threads, and the
95th percentile rises from 50 to 52 ms; the fusion waits 0.1 ms a sweep either way.

```sh
docker compose run --rm pipes run --consumer-delay-ms 100 --cap 1 --pair-wait-ms 300
```

The slow-camera experiment with two cameras. The left detector sleeps an extra 100 ms
per frame and its one-frame queue drops frames (40 of 154 on this run); the right one
is not slowed and drops none. With one camera those sweeps expired, 38 of 154 on one
run. With two, all 154 are answered: each sweep whose left frame was lost is
built on the right frame instead, the picture's corner says `RIGHT camera`, every
match in it is one camera's (`one_camera`, which the headline does not repeat), and
`summary.json` counts them (`sets_from_right_camera`). The wait is the same 103 ms
per sweep the one-camera run paid, because the fusion cannot know a frame is lost
until the next one arrives.

```sh
docker compose run --rm pipes run --consumer-delay-ms 2000
docker compose run --rm pipes run --right-delay-ms 2000
```

A camera that breaks. A detector slowed to two seconds a frame is, for every
purpose, a dead stream: it finishes 9 of 154 frames, each far too old to pair. With
the left one broken, the fusion waits for it on three sweeps, declares it silent,
and builds every set from the right frame without waiting: all 154 sweeps are
answered, the lidar chain loses nothing (`obj->track` evicts 0), 404 matches are
fused from the right camera alone, and the most urgent answer is the same car as in
the healthy run, with `RIGHT camera, left camera silent` beside it. Each of the nine
frames that do
arrive is a sign of life, so the three sweeps after it wait for the left camera
again (and are demoted, their matches one camera's) before it falls silent once
more: over the run the left camera is silent on 130 sweeps, 24 are demoted, and the
fusion waits 16 ms a sweep instead of 103. The second command (a hidden test hook)
breaks the right detector instead, and the left camera carries on alone the same
way: 422 matches, every answer saying `right camera silent`. These figures are from
runs streaming to a live viewer.

## The detector on the GPU, and a bigger model

```sh
docker compose run --rm pipes-gpu                                   # YOLOX-L and the lidar detector, on the GPU
docker compose run --rm pipes-gpu run --detector-model yolox_s-1024 --detector-backend cuda
docker compose run --rm pipes run --detector-backend ort            # ONNX Runtime on the CPU
docker compose run --rm pipes bench-detector --model yolox_s-1024 --backend ort --threads 4
```

`pipes bench-detector` times one model on one runtime over the first 40 frames of a
drive, outside the pipeline, and runs them twice to say whether the answer repeats.
Measured on this host (i7-12700, RTX 3060), in milliseconds per frame at the median,
the letterbox and the decode included:

| Model (input) | Parameters | Detections per frame | tract, 1 thread | ONNX Runtime, 4 threads | CUDA | CUDA, TF32 |
|---|---|---|---|---|---|---|
| YOLOX-Nano (640x192) | 0.9 M | 6.5 | 33 | 9.5 | 7.2 | 7.2 |
| YOLOX-S (640x192) | 9 M | 7.2 | 104 | 25 | 13 | |
| YOLOX-S (1024x320) | 9 M | 10.9 | 286 | 68 | 23 | 14 |
| YOLOX-M (1024x320) | 25 M | 12.6 | 734 | 164 (8 threads) | 52 (32 with `--cuda-search default`) | 21 |

Every cell repeated byte for byte on a second pass, and the CUDA cells on a second
process. ONNX Runtime on one CPU thread already runs Nano in 15 ms against tract's
33, with the same detections to the byte; four threads bring it to 9.5 ms and eight
add nothing. The GPU's time is almost all the network: the letterbox alone is 4 ms
at 640x192 and 7 ms at 1024x320, on the CPU either way. cuDNN's algorithm choice
matters: its default algorithm runs Nano in 168 ms (the depthwise convolutions) and
YOLOX-M in 32 ms, its heuristic choice runs them in 7 and 52, so a run uses the
heuristic, which is the same choice on every run, and records it; `--cuda-search`
and `--cuda-tf32` (hidden) change it, and TF32 halves the bigger models at the price
of rounding the matmuls' inputs. The bigger models earn their time in what they see:
YOLOX-M at 1024x320 finds 12.6 objects a frame where Nano finds 6.5.

In the pipeline the picture changes: a camera stage takes about twice its time on the
bench, so YOLOX-M's 52 ms becomes 91 ms a stage and Nano's 7 becomes 27. That was
first put down to the two cameras' sessions sharing the GPU.
[The lidar detector](#the-lidar-detector)'s runs say otherwise: a third session on the
same GPU makes every stage faster, so the likelier cause is a GPU that idles between
frames. These are the two-camera runs of drive 0005 with the viewer on, the camera
stage's time per frame, when its result was ready against the sweep's end (the diamond
on the Demo page), and what it found over the run:

| Model, runtime | Stage ms p50 / p95 | Camera done after sweep end | Detections | Confirmed by both cameras |
|---|---|---|---|---|
| Nano, tract (the default) | 41 / 48 | +0.1 ms | 1,227 | 393 |
| Nano, ONNX Runtime CPU, 4 threads | 17 / 20 | -25 ms | 1,227 | 393 |
| Nano, CUDA | 27 / 35 | -14 ms | 1,227 | 393 |
| S at 1024x320, CUDA | 46 / 54 | +5 ms | 1,852 | 435 |
| S at 1024x320, CUDA, TF32 | 36 / 51 | -5 ms | 1,852 | 435 |
| M at 1024x320, CUDA | 91 / 94 | +50 ms (the fusion waits 41 ms a sweep) | 2,003 | 476 |
| M at 1024x320, CUDA, `--cuda-search default` | 53 / 57 | +12 ms | 2,003 | 476 |
| M at 1024x320, CUDA, TF32 | 46 / 55 | +5 ms | 2,003 | 476 |

Every run answered all 154 sweeps and held every invariant, and the three runtimes
find the same 1,227 Nano detections. The cheapest improvement is ONNX Runtime on the
CPU: the same detector, ready 25 ms before the sweep ends instead of at it. The
biggest model on the GPU finds 63 % more objects a frame than Nano and has both
cameras confirm 21 % more tracks, and with TF32 its result is ready where Nano's is
today. That made it the GPU image's model until [bigger ones](#the-biggest-models)
were measured against KITTI's labels. Without TF32 it needs
cuDNN's default algorithm (`--cuda-search default`) to be ready in time, and on the
heuristic it is 50 ms late every sweep, which the fusion waits out: the answer still
comes, 41 ms a sweep later. One batched session for both cameras, instead of two
taking turns, was the step planned next; the lidar detector made it unnecessary, as
the next section shows.

## The lidar detector

```sh
docker compose run --rm pipes-gpu                                                  # lidar detector and YOLOX-L, all on the GPU
docker compose run --rm pipes-gpu run --lidar-detector on --detector-backend ort   # the cameras on the CPU instead
docker compose run --rm pipes-gpu run --lidar-detector on --lidar-delay-ms 2000    # a lidar detector that breaks
docker compose run --rm pipes-gpu bench-lidar --views ahead                        # the network alone, one pass
```

`--lidar-detector on` runs PointPillars on every sweep and names the tracks it finds.
This section is the detector as first built and measured: 32-bit weights, YOLOX-M
beside it, and one pass a sweep over the 69 m ahead (`--lidar-views ahead`).
[The biggest models](#the-biggest-models) has what the GPU image runs now.
On drive 0005 it reports 6,959 boxes, 45 a sweep. 5,203 of them score under 0.3 and
name nothing: at the network's own threshold of 0.1 it calls lamp posts pedestrians.
The rest name 1,407 of the drive's 23,821 track observations: 936 as cars, 260 as
pedestrians and 211 as cyclists. Where a camera named the same track the two agree, on
all 352 with YOLOX-Nano and on 400 of 402 with YOLOX-M, and the lidar alone names
1,055. On drive 0009 it names 7,351 of 63,787, 6,940 of them cars, and agrees with the
camera on 1,484 of 1,486.

The plan was to keep the cameras on the CPU so that they would not compete with the
lidar detector for the GPU. Measured on this host (i7-12700, RTX 3060, under WSL), it
is the other way round. These are runs of drive 0005 with the viewer on and the lidar
detector on the GPU in all but the first:

| Cameras | Camera stage ms p50 / p95 | Camera done after sweep end | Lidar detector stage ms p50 / p95 | Fusion waits for the boxes | Answer after sweep end |
|---|---|---|---|---|---|
| Nano, tract, no lidar detector | 41 / 52 | +0.2 ms | | | +10 ms |
| Nano, tract | 40 / 49 | -1 ms | 39 / 45 | 24 ms a sweep | +40 ms |
| Nano, ONNX Runtime on the CPU | 15 / 19 | -26 ms | 38 / 44 | 23 ms | +38 ms |
| Nano, CUDA | 12 / 17 | -29 ms | 34 / 37 | 20 ms | +34 ms |
| S at 1024x320, CUDA, TF32 | 24 / 28 | -17 ms | 27 / 29 | 16 ms | +28 ms |
| M at 1024x320, CUDA, TF32 | 33 / 37 | -8 ms | 21 / 22 | 11 ms | +21 ms |

Every run answered all 154 sweeps and held every invariant. The lidar detector cannot
start before its sweep ends, so its boxes are the last thing the fusion waits for and
the answer comes when they do, 21 to 40 ms after the sweep's end instead of 10. The
cameras are not what it waits for in any row. And each network runs faster the more
the GPU has to do: PointPillars' network takes 33 to 36 ms a sweep when it is the GPU's
only work, 31 with YOLOX-Nano beside it and 18 with YOLOX-M, which itself takes 33 ms
a frame beside the lidar detector against 47 alone. Back to back, with no gaps,
`bench-lidar` measures the network at 19 ms. The likeliest reason is power management:
a GPU that idles between sweeps slows its clocks, and a busy one does not. The two
also overlap little: the cameras fire in the middle of a sweep and are done before it
ends, which is when the lidar detector starts. So the GPU image runs all three sessions
on the GPU, and one batched session for both cameras would gain nothing: nothing waits
for the cameras. On another machine the balance may
differ, and the second command above is the plan as first made, the cameras on the CPU.

A lidar detector slowed to two seconds a sweep (the third command, a hidden test hook)
finishes 10 of the 154 sweeps. The fusion waits for it on 24 sweeps, declares it silent
on the other 130, and answers all 154 with the cameras' classes alone, 16 ms a sweep
later on average; the picture's corner says `lidar detector silent`. On the CPU
(`--lidar-backend ort`) the network takes 235 ms a sweep, longer than a sweep lasts,
so at real time most boxes come late and the run says so; the tests run it that way,
with the replay slowed to a quarter and the fusion's wait raised to 400 ms. In a
container that was not handed a GPU, a `cuda` backend stops before the run starts and
says which command gives it one.

## The biggest models

```sh
docker compose run --rm pipes-gpu                                             # YOLOX-L on both cameras, the lidar detector looking both ways
docker compose run --rm pipes-gpu run --lidar-detector on --detector-backend cuda --detector-model yolox_x-1248   # YOLOX-X, the biggest
docker compose run --rm pipes-gpu run --lidar-detector on --detector-backend cuda --detector-model yolox_l-1248 --lidar-views ahead   # the lidar detector looking ahead only
docker compose run --rm score runs/<name> 2011_09_26_drive_0005_sync          # a run's detectors against KITTI's labels
```

How big a model can each sensor carry at 10 Hz on this GPU, and is bigger better? Size
is easy to measure. Better needed a witness the pipeline never reads, and KITTI has
one: for these four drives its labellers drew a 3D box round every road user in every
frame. `fetch` downloads those labels beside each drive, and `score` reads what a
run's detectors wrote and counts how many labelled road users each found: for the
camera a detection overlapping the object's rectangle in the image by half (IoU 0.5;
0.3 for a cyclist, whom the camera detector sees as a person and a bicycle), for the
lidar a box scoring 0.3 or more overlapping its footprint on the ground. Only objects
marked visible or partly occluded, within 50 m, count.

The cameras, on all four drives, 767 frames holding 2,351 labelled road users (2,123
vehicles, 74 pedestrians, 154 cyclists):

| Camera model (input) | Parameters | Found | Vehicles | Pedestrians | Cyclists | Detections under no label, a frame |
|---|---|---|---|---|---|---|
| YOLOX-Nano (640x192), the plain run's default | 0.9 M | 91.5 % | 92.6 % | 44.6 % | 98.7 % | 1.7 |
| YOLOX-M (1024x320), the GPU image's default before | 25 M | 97.0 % | 97.6 % | 74.3 % | 100 % | 4.3 |
| YOLOX-L (1248x384, 16-bit), the GPU image's default | 54 M | 97.4 % | 98.1 % | 75.7 % | 99.4 % | 5.2 |
| YOLOX-X (1248x384, 16-bit) | 99 M | 97.5 % | 98.0 % | 78.4 % | 100 % | 5.2 |

And what each costs in the pipeline: drive 0005 with the viewer on, the lidar detector
on the GPU beside the cameras, every run answering all 154 sweeps with every invariant
held:

| Cameras | Lidar detector looks | Camera stage ms p50 / p95 | Lidar detector stage ms p50 / p95 | Answer after sweep end, ms p50 / p95 |
|---|---|---|---|---|
| YOLOX-M (TF32) | ahead and behind | 32 / 34 | 31 / 34 | +32 / +34 |
| YOLOX-L | ahead | 47 / 51 | 18 / 20 | +18 / +21 |
| YOLOX-L (the default) | ahead and behind | 47 / 51 | 34 / 38 | +35 / +39 |
| YOLOX-X | ahead and behind | 77 / 87 | 61 / 65 | +62 / +66 |

Bigger is better up to YOLOX-L, by less at each step, and not past it. The step that
matters is the first: YOLOX-M finds 97.6 % of the labelled vehicles and three
pedestrians in four, where Nano finds 92.6 % and fewer than half. YOLOX-L at the
frame's own resolution adds half a point of vehicles, about ten of them, and its 15 ms
of extra camera time costs the answer 3 ms, because the answer waits for the lidar
detector's boxes either way. YOLOX-X has almost twice the parameters again and finds
two more pedestrians and one more cyclist of 228, and no more vehicles. Its cost is not
small: each camera stage takes 77 ms, the cameras are still on the GPU when the sweep
ends, so the lidar detector's network takes 54 ms instead of 27, the fusion waits 29 ms
a sweep for the cameras, and the answer comes 62 ms after the sweep's end instead of 35.
YOLOX-X runs, and it is one flag away, but the GPU image's default is the biggest model
that is also the best.

What let a 54-million-parameter network in at all is half precision. The files the GPU
loads store their weights as 16-bit floats (`scripts/prepare_models.py` converts them,
and the result is pinned by its hash like every other model). Measured outside the
pipeline on this GPU, YOLOX-L's network at 1248x384 takes 31 ms a frame with 32-bit
weights and 19 ms with 16-bit, and on these drives the two find the same labelled
objects. `pipes bench-detector --model yolox_l-1248 --backend cuda` times the 16-bit
file at 20 ms for the network and its decode, and 10 ms more for the letterbox, which
is on the CPU. The
bigger models also find more that no label stands under, 5 detections a frame against
Nano's 2: the labels stop at about 60 m and at road users, and a bigger network sees
further and names more kinds of thing.

The lidar has no bigger model to move to. The detectors that beat PointPillars on
KITTI are built on sparse 3D convolutions, which ONNX Runtime cannot run, and the one
bigger network published as plain ONNX was trained on another lidar. So the same
frozen weights were pushed two other ways. As 16-bit floats the network takes 14 ms a
pass instead of 18. And it runs twice a sweep, the second time on the sweep turned half
round, so what is behind the car is named as well; overlapping boxes from the two
passes are removed together, so a car beside the car is one box. Against the labels,
over 763 sweeps:

| Lidar detector, boxes scoring 0.3 or more | Found |
|---|---|
| Vehicles ahead | 90.8 % of 2,200 |
| Pedestrians ahead | 86.5 % of 74 |
| Cyclists ahead | 99.4 % of 154 |
| Ahead, by range: 0 to 15 m, 15 to 30 m, 30 to 50 m | 95.9 %, 89.4 %, 85.2 % |
| Boxes in the camera's view under no label | 2.5 a sweep |
| Parked cars behind the car: 0 to 15 m, 15 to 30 m, 30 to 50 m | 92.7 % of 769, 73.4 % of 1,497, 25.1 % of 1,944 |

The drives differ: the lidar detector finds 98.6 % and 99.6 % of the labelled vehicles
on drives 0009 and 0013 and 47 % on drive 0005, where most of what it misses is beyond
15 m; why that street is hard for it has not been looked into. Nothing behind the car
is labelled, because the camera does not see it, so the last row is measured by
carrying: a parked car stays where it was labelled, and its box, moved
through the car's own GPS/IMU motion into the sweeps after the car has passed it, is
ground truth behind the car. Where the labels go on, the carried box overlaps the
labelled one by 0.88 at the median. The far figures are lower bounds, since a parked
car 40 m behind is mostly hidden by the ones nearer.

Looking behind doubles what the lidar detector names, 2,666 track observations on drive
0005 against 1,404, and costs 17 ms on the answer, which waits for the boxes. Nothing
behind the car can be in its path, so `--lidar-views ahead` gives that time back when
only the path ahead matters.

The dashboard shows what the lidar detector says without a new view: its boxes scoring
0.3 or more in the lidar views, in violet with class and score, all the way round the
car; its class on every track in the picture it named; and in the picture's corner,
after the paired frame, its tally for the sweep, such as `lidar: 2 cars, 2 pedestrians,
2 cyclists`.

## The decision

Removed on 2026-10-13: on these drives the rule said `continue` almost everywhere,
and most of its few `slow`s were for parked cars beside a bend. The answer keeps the
time to contact; `decisions.csv` and the `check` service went with the rule. This is
the record of what it did.

```sh
docker compose run --rm pipes run --name decided
docker compose run --rm check runs/decided 2011_09_26_drive_0005_sync
```

Every run that reaches the answer decides, and prints the decision second by second:

```text
DECISION: continue on 129 answers, slow on 25, brake on 0 | 154 of the drive's 154 sweeps have a decision, ...
  second  10: continue: pedestrian or cyclist #110 12.7 m along the path, closing at least 2.7 m/s (3.0 measured), needs 0.47 m/s^2, thresholds halved  [sweep 106; 10 answers: 10 continue]
  second  11: slow: pedestrian or cyclist #110 13.0 m along the path, closing at least 3.0 m/s (3.3 measured), needs 0.57 m/s^2, thresholds halved  [sweep 112; 10 answers: 2 continue, 8 slow]
  second  12: slow: vehicle #2329 12.8 m along the path, closing at least 4.0 m/s (5.5 measured), needs 1.05 m/s^2  [sweep 125; 9 answers: 1 continue, 8 slow]
```

A second's line is its most demanding answer, with how the second's other answers
split. `decisions.csv` has every sweep's row: the gap, the closing speed as measured,
how much of it is established, the deceleration needed, what the sensors call the
object and why the thresholds were halved.

A recorded drive also says what happened next, in the GPS/IMU rows after each sweep,
and the pipeline reads none of them. The second command, `scripts/check_decisions.py`
in an image of its own, uses them as a witness: for each sweep, did the car go on to
drive over the place of the track that decided, or pass beside it? These are the four
drives with the GPU image's default run, counting the sweeps whose own verdict asked
for slowing or braking, before the three-sweep hold:

| Drive | Sweeps | Asked `slow` | Deciding track on the driven path | Beside it | Drive ended first | Asked `brake` | With velocities as measured: `slow` / `brake` |
|---|---|---|---|---|---|---|---|
| 0005 | 154 | 25 | 15 | 6 | 4 | 0 | 45 / 4 |
| 0009 | 443 | 9 | 0 | 9 | 0 | 0 | 28 / 0 |
| 0013 | 144 | 0 | | | | 0 | 18 / 29 |
| 0048 | 22 | 0 | | | | 0 | 2 / 0 |

No driver brakes on these drives, and the rule asks for braking on none. The last
column is the same rule on the tracks' velocities as measured, which is why it does
not use them: the 29 on drive 0013 are one vehicle ahead, pulling away faster than the
tracker's 50 km/h bound, lost and found again on every sweep, each new track starting
with a velocity made of where its cluster happened to form.

On drive 0005 the 15 are a cyclist the car follows round a bend, 13 m ahead and
closing at 3 m/s; a cyclist halves the thresholds, so 0.5 m/s² is enough to ask. The
driver did not slow. The 6, and the 9 on drive 0009, are cars parked on the inside of
a bend the car was coming out of: the arc of the moment runs through them, the car
straightens, and passes them by more than 2 m. That is the limit of predicting the
path from one instant, and it can be measured without any detector. This is the share
of each drive's frames whose position after so many metres was within half the car's
width of where the arc had put it, and the same for a straight line:

| Drive | Path | 5 m | 10 m | 15 m | 20 m | 30 m |
|---|---|---|---|---|---|---|
| 0005 | arc | 100 % | 79 % | 57 % | 28 % | 1 % |
| | straight | 95 % | 12 % | 7 % | 2 % | 0 % |
| 0009 | arc | 98 % | 86 % | 78 % | 70 % | 59 % |
| | straight | 88 % | 82 % | 79 % | 75 % | 46 % |
| 0013 | arc | 100 % | 100 % | 100 % | 92 % | 69 % |
| | straight | 100 % | 100 % | 100 % | 92 % | 56 % |

The arc is the road for the first ten metres and, on a street that keeps turning, a
guess past twenty. A default run of drive 0005, without the lidar detector, asks for
slowing on 19 sweeps and for braking on none.
