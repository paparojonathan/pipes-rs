#!/usr/bin/env python3
"""Score a run's two detectors against KITTI's own labels.

    docker compose run --rm score runs/<name> <drive> [runs/<name> <drive> ...]
    python3 scripts/score_detectors.py runs/<name> <kitti root>/<date>/<drive> ...

The first form needs nothing installed: the `score` service has Python and
the two packages, and compose mounts `runs/` and the KITTI data. `<drive>` is
a drive's name, such as 2011_09_26_drive_0005_sync, found under
`PIPES_KITTI_ROOT`, or the path of its folder. Several runs, each with its
drive, are scored together.

KITTI publishes, for some raw drives, the 3D box of every road user a person
labelled in every frame (`tracklet_labels.xml`, fetched with the drive). The
pipeline never reads them, so they are a witness: this script reads what a
run's detectors wrote (`cam_det.arrows`, `pillar_det.arrows`) and counts how
many labelled objects each found.

What is counted. A labelled car, van, truck, tram, pedestrian (sitting or
not) or cyclist that the labeller marked visible or partly occluded and at
least partly inside the image, and that lies within 50 m ahead. For the
camera it must also be 25 px tall or more in the image.

Found. By the camera: a detection of a road-user class overlaps the object's
rectangle in the image by IoU 0.5 or more; 0.3 for a cyclist, because the
label is one box round rider and bicycle and the camera detector, trained on
COCO, gives a person and a bicycle. By the lidar: a box scoring 0.3 or more
overlaps its footprint on the ground by IoU 0.5 for a vehicle, 0.25 for a
pedestrian or a cyclist (the KITTI benchmark's own bars are 0.7 and 0.5, in
3D).

Unmatched. Detections inside the camera's view that overlap no labelled
object (IoU under 0.3 in the image, under 0.1 on the ground). An upper bound on false alarms, not a count of them: the
labels stop where the labeller did, at about 60 m and at what the camera
shows.

Behind the vehicle. Nothing there is labelled, because the camera does not
see it. But a parked car stays where it was labelled: carried through the
vehicle's own motion (the GPS/IMU rows) into the sweeps after the vehicle
has passed it, its box is ground truth behind the vehicle. A run whose lidar
detector looked behind is scored on those, where a box scoring 0.3 or more
overlaps the carried footprint by IoU 0.5. A lower bound: whether the parked
car could still be seen from there is not known.

Needs `numpy` and `pyarrow` (pip install numpy pyarrow).
"""
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.ipc as ipc

IMG_W, IMG_H = 1242.0, 375.0
EARTH_RADIUS_M = 6378137.0
# The lidar's origin in the GPS/IMU's frame, metres (`calib_imu_to_velo.txt`).
LIDAR_AHEAD_M, LIDAR_LEFT_M = 0.81, -0.32
GROUP = {"Car": "vehicle", "Van": "vehicle", "Truck": "vehicle", "Tram": "vehicle",
         "Pedestrian": "pedestrian", "Person (sitting)": "pedestrian", "Cyclist": "cyclist"}
GROUPS = ["vehicle", "pedestrian", "cyclist"]
# COCO ids of the classes a road user can be given: person, bicycle, car,
# motorcycle, bus, train, truck.
ROAD_USERS = {0, 1, 2, 3, 5, 6, 7}
MIN_HEIGHT_PX = 25.0
CAMERA_IOU = {"vehicle": 0.5, "pedestrian": 0.5, "cyclist": 0.3}
MAX_RANGE_M = 50.0
LIDAR_SCORE = 0.3
LIDAR_IOU = {"vehicle": 0.5, "pedestrian": 0.25, "cyclist": 0.25}
RANGES = [(0, 15), (15, 30), (30, 50)]


def read_calib(date_dir):
    def table(path):
        out = {}
        for line in open(path):
            key, _, values = line.partition(":")
            try:
                out[key.strip()] = np.array([float(v) for v in values.split()])
            except ValueError:
                pass
        return out

    cam = table(os.path.join(date_dir, "calib_cam_to_cam.txt"))
    velo = table(os.path.join(date_dir, "calib_velo_to_cam.txt"))
    rect = np.eye(4)
    rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
    to_cam = np.eye(4)
    to_cam[:3, :3] = velo["R"].reshape(3, 3)
    to_cam[:3, 3] = velo["T"]
    return cam["P_rect_02"].reshape(3, 4), rect @ to_cam


def read_labels(path):
    """frame -> the objects labelled in it, in the lidar's frame; z is the box's bottom."""
    frames = {}
    root = ET.parse(path).getroot().find("tracklets")
    for track, item in enumerate(root.findall("item")):
        h, w, l = (float(item.findtext(k)) for k in ("h", "w", "l"))
        first = int(item.findtext("first_frame"))
        for k, pose in enumerate(item.find("poses").findall("item")):
            frames.setdefault(first + k, []).append(dict(
                type=item.findtext("objectType"), track=track, l=l, w=w, h=h,
                x=float(pose.findtext("tx")), y=float(pose.findtext("ty")), z=float(pose.findtext("tz")),
                yaw=float(pose.findtext("rz")), occlusion=int(pose.findtext("occlusion")),
                truncation=int(pose.findtext("truncation"))))
    return frames


def counted(o):
    return (o["type"] in GROUP and o["occlusion"] in (0, 1) and o["truncation"] in (0, 1)
            and 0.0 < o["x"] and math.hypot(o["x"], o["y"]) < MAX_RANGE_M)


def footprint(x, y, dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + u * c - v * s, y + u * s + v * c)
            for u, v in ((-dx / 2, -dy / 2), (dx / 2, -dy / 2), (dx / 2, dy / 2), (-dx / 2, dy / 2))]


def image_box(o, calib):
    """The object's rectangle in the left colour image, or None when it is not wholly ahead of the camera."""
    project, to_cam = calib
    corners = [(px, py, o["z"] + dz, 1.0) for dz in (0.0, o["h"])
               for px, py in footprint(o["x"], o["y"], o["l"], o["w"], o["yaw"])]
    cam = (to_cam @ np.array(corners).T).T
    if np.any(cam[:, 2] < 0.5):
        return None
    uv = (project @ cam.T).T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    box = (max(u.min(), 0.0), max(v.min(), 0.0), min(u.max(), IMG_W), min(v.max(), IMG_H))
    return box if box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0 else None


def in_view(calib, x, y, z):
    project, to_cam = calib
    cam = to_cam @ np.array([x, y, z, 1.0])
    if cam[2] < 1.0:
        return False
    u, v, w = project @ cam
    return 0.0 <= u / w <= IMG_W and 0.0 <= v / w <= IMG_H


def iou_image(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def clip(subject, clipper):
    """`subject` cut to the convex, counter-clockwise `clipper` (Sutherland-Hodgman)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def crossing(a, b, p, q):
        d1, d2 = (b[0] - a[0], b[1] - a[1]), (q[0] - p[0], q[1] - p[1])
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return p
        t = ((p[0] - a[0]) * d2[1] - (p[1] - a[1]) * d2[0]) / den
        return (a[0] + t * d1[0], a[1] + t * d1[1])

    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        points, out = out, []
        if not points:
            break
        last = points[-1]
        for p in points:
            if inside(p, a, b):
                if not inside(last, a, b):
                    out.append(crossing(a, b, last, p))
                out.append(p)
            elif inside(last, a, b):
                out.append(crossing(a, b, last, p))
            last = p
    return out


def iou_ground(a, b):
    """Overlap on the ground of two (x, y, length, width, heading) rectangles, turned as they are."""
    poly = clip(footprint(*a), footprint(*b))
    n = len(poly)
    inter = abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2 if n >= 3 else 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 1e-9 else 0.0


def batches(path):
    with open(path, "rb") as f:
        reader = ipc.open_stream(f)
        meta = {k.decode(): v.decode() for k, v in (reader.schema.metadata or {}).items()}
        return meta, [b.to_pydict() for b in reader]


class Tally:
    """Found and counted, by group and by a second key."""

    def __init__(self, keys):
        self.group = {g: [0, 0] for g in GROUPS}
        self.key = {k: [0, 0] for k in keys}

    def add(self, group, key, found):
        for cell in (self.group[group], self.key.get(key)):
            if cell is not None:
                cell[0] += int(found)
                cell[1] += 1

    def all(self):
        return [sum(v[0] for v in self.group.values()), sum(v[1] for v in self.group.values())]


def share(cell):
    return f"{100.0 * cell[0] / cell[1]:5.1f}% of {cell[1]}" if cell[1] else "none labelled"


def score_camera(run, labels, calib, out):
    """One run's camera detections against the labels' rectangles in the image."""
    path = os.path.join(run, "cam_det.arrows")
    if not os.path.exists(path):
        return None
    meta, rows = batches(path)
    for row in rows:
        frame = row["frame_seq"][0]
        dets = [d for d in row["detections"][0] if d["class_id"] in ROAD_USERS]
        boxes = [(d["x0"], d["y0"], d["x1"], d["y1"]) for d in dets]
        labelled = [(o, image_box(o, calib)) for o in labels.get(frame, []) if o["truncation"] in (0, 1)]
        labelled = [(o, b) for o, b in labelled if b is not None]
        used = set()
        for o, box in labelled:
            height = box[3] - box[1]
            if not counted(o) or height < MIN_HEIGHT_PX:
                continue
            best = max(((iou_image(b, box), j) for j, b in enumerate(boxes)), default=(0.0, -1))
            key = "25-40 px" if height < 40 else ("40-80 px" if height < 80 else "80 px and taller")
            found = best[0] >= CAMERA_IOU[GROUP[o["type"]]]
            out["tally"].add(GROUP[o["type"]], key, found)
            if found:
                used.add(best[1])
        out["frames"] += 1
        out["detections"] += len(boxes)
        out["unmatched"] += sum(1 for j, b in enumerate(boxes)
                                if j not in used and all(iou_image(b, box) < 0.3 for _, box in labelled))
    return meta


def score_lidar(run, labels, calib, out):
    """One run's lidar boxes against the labels' footprints, ahead of the vehicle. Returns the boxes by sweep."""
    path = os.path.join(run, "pillar_det.arrows")
    if not os.path.exists(path):
        return None, {}
    meta, rows = batches(path)
    by_sweep = {}
    for row in rows:
        sweep = row["source_sweep_seq"][0]
        dets = [d for d in row["detections"][0] if d[8] >= LIDAR_SCORE]
        by_sweep[sweep] = dets
        labelled = [o for o in labels.get(sweep, []) if o["truncation"] in (0, 1) and o["x"] > 0.0]
        used = set()
        for o in labelled:
            if not counted(o):
                continue
            group = GROUP[o["type"]]
            box = (o["x"], o["y"], o["l"], o["w"], o["yaw"])
            best = max(((iou_ground((d[0], d[1], d[3], d[4], d[6]), box), j) for j, d in enumerate(dets)
                        if abs(d[0] - o["x"]) < 6 and abs(d[1] - o["y"]) < 6), default=(0.0, -1))
            rng = math.hypot(o["x"], o["y"])
            key = next((r for r in RANGES if r[0] <= rng < r[1]), None)
            out["tally"].add(group, key, best[0] >= LIDAR_IOU[group])
            if best[0] >= LIDAR_IOU[group]:
                used.add(best[1])
        out["sweeps"] += 1
        for j, d in enumerate(dets):
            if math.hypot(d[0], d[1]) >= MAX_RANGE_M or not in_view(calib, d[0], d[1], d[2]):
                continue
            out["in_view"] += 1
            near = [o for o in labelled if abs(d[0] - o["x"]) < 8 and abs(d[1] - o["y"]) < 8]
            if j not in used and all(iou_ground((d[0], d[1], d[3], d[4], d[6]), (o["x"], o["y"], o["l"], o["w"], o["yaw"])) < 0.1 for o in near):
                out["unmatched"] += 1
    return meta, by_sweep


def lidar_poses(drive_dir):
    """The lidar's position and heading at every frame, on a flat map around the drive's start."""
    rows = np.array([[float(v) for v in open(p).read().split()] for p in sorted(glob.glob(os.path.join(drive_dir, "oxts", "data", "*.txt")))])
    if len(rows) == 0:
        return None
    scale = math.cos(math.radians(rows[0, 0]))
    east = scale * np.radians(rows[:, 1]) * EARTH_RADIUS_M
    north = scale * EARTH_RADIUS_M * np.log(np.tan(np.radians(90.0 + rows[:, 0]) / 2.0))
    yaw = rows[:, 5]
    return (east + np.cos(yaw) * LIDAR_AHEAD_M - np.sin(yaw) * LIDAR_LEFT_M,
            north + np.sin(yaw) * LIDAR_AHEAD_M + np.cos(yaw) * LIDAR_LEFT_M, yaw)


def score_behind(labels, poses, by_sweep, out):
    """Standing vehicles carried behind the vehicle, against the boxes found there."""
    east, north, yaw = poses
    n = len(east)

    def to_map(k, x, y, a):
        c, s = math.cos(yaw[k]), math.sin(yaw[k])
        return east[k] + c * x - s * y, north[k] + s * x + c * y, a + yaw[k]

    def to_lidar(k, mx, my, ma):
        c, s = math.cos(-yaw[k]), math.sin(-yaw[k])
        dx, dy = mx - east[k], my - north[k]
        return c * dx - s * dy, s * dx + c * dy, ma - yaw[k]

    tracks = {}
    for frame, objs in labels.items():
        for o in objs:
            tracks.setdefault(o["track"], []).append((frame, o))
    for seen in tracks.values():
        seen.sort(key=lambda t: t[0])
        if GROUP.get(seen[0][1]["type"]) != "vehicle" or len(seen) < 5 or seen[-1][0] >= n:
            continue
        on_map = [to_map(f, o["x"], o["y"], o["yaw"]) for f, o in seen]
        if max(math.hypot(m[0] - on_map[0][0], m[1] - on_map[0][1]) for m in on_map) > 0.5:
            continue  # it moved
        # Where the labels still say where it is, the carrying is checked against them.
        first = seen[0][1]
        for (frame, o), _ in zip(seen[1:], on_map[1:]):
            x, y, a = to_lidar(frame, *on_map[0])
            out["carried"].append(iou_ground((x, y, first["l"], first["w"], a), (o["x"], o["y"], o["l"], o["w"], o["yaw"])))
        last_frame, last = seen[-1]
        for frame in range(last_frame + 1, min(n, last_frame + 150)):
            if frame not in by_sweep:
                continue
            x, y, a = to_lidar(frame, *on_map[-1])
            rng = math.hypot(x, y)
            key = next((r for r in RANGES if r[0] <= rng < r[1]), None)
            if not (-MAX_RANGE_M < x < -1.0 and abs(y) < 39.0) or key is None:
                continue
            box = (x, y, last["l"], last["w"], a)
            best = max((iou_ground((d[0], d[1], d[3], d[4], d[6]), box) for d in by_sweep[frame]
                        if abs(d[0] - x) < 6 and abs(d[1] - y) < 6), default=0.0)
            out["tally"].add("vehicle", key, best >= 0.5)


def main(args):
    if len(args) < 2 or len(args) % 2:
        raise SystemExit(__doc__)
    cam = dict(tally=Tally(["25-40 px", "40-80 px", "80 px and taller"]), frames=0, detections=0, unmatched=0)
    lidar = dict(tally=Tally(RANGES), sweeps=0, in_view=0, unmatched=0)
    behind = dict(tally=Tally(RANGES), carried=[])
    cam_meta = lidar_meta = None
    looked_behind = False
    for run, drive in zip(args[0::2], args[1::2]):
        drive_dir = drive if os.path.isdir(drive) else os.path.join(os.environ.get("PIPES_KITTI_ROOT", "../data/kitti"), drive[:10], drive)
        label_file = os.path.join(drive_dir, "tracklet_labels.xml")
        if not os.path.exists(label_file):
            raise SystemExit(f"{label_file} not found: KITTI publishes labels for some raw drives only; `docker compose run --rm fetch` fetches the ones it has")
        labels = read_labels(label_file)
        calib = read_calib(os.path.dirname(os.path.normpath(drive_dir)))
        cam_meta = score_camera(run, labels, calib, cam) or cam_meta
        meta, by_sweep = score_lidar(run, labels, calib, lidar)
        lidar_meta = meta or lidar_meta
        poses = lidar_poses(drive_dir)
        if meta and "behind" in meta.get("views", "") and poses is not None:
            looked_behind = True
            score_behind(labels, poses, by_sweep, behind)
        print(f"{run} against {os.path.basename(os.path.normpath(drive_dir))}: {sum(len(v) for v in labels.values())} labelled boxes in {len(labels)} frames")

    if cam_meta:
        t = cam["tally"]
        print(f"camera detector {cam_meta.get('model')} ({cam_meta.get('input')}), {cam['frames']} frames:")
        print(f"  found {share(t.all())} | " + " | ".join(f"{g}s {share(t.group[g])}" for g in GROUPS))
        print("  by height in the image: " + " | ".join(f"{k} {share(v)}" for k, v in t.key.items()))
        print(f"  detections of road users {cam['detections'] / max(cam['frames'], 1):.2f} a frame, overlapping no label {cam['unmatched'] / max(cam['frames'], 1):.2f} a frame")
    if lidar_meta:
        t = lidar["tally"]
        print(f"lidar detector {lidar_meta.get('model')} on {lidar_meta.get('backend')}, looking {lidar_meta.get('views', 'ahead')}, boxes scoring {LIDAR_SCORE} or more, {lidar['sweeps']} sweeps:")
        print(f"  found {share(t.all())} | " + " | ".join(f"{g}s {share(t.group[g])}" for g in GROUPS))
        print("  by range: " + " | ".join(f"{k[0]}-{k[1]} m {share(v)}" for k, v in t.key.items()))
        print(f"  boxes in the camera's view within {MAX_RANGE_M:.0f} m {lidar['in_view'] / max(lidar['sweeps'], 1):.2f} a sweep, overlapping no label {lidar['unmatched'] / max(lidar['sweeps'], 1):.2f} a sweep")
    if looked_behind:
        t, carried = behind["tally"], np.array(behind["carried"])
        print("behind the vehicle, standing vehicles carried there from their labels:")
        print(f"  found {share(t.all())} | " + " | ".join(f"{k[0]}-{k[1]} m {share(v)}" for k, v in t.key.items()))
        if len(carried):
            print(f"  the carrying itself, where the labels go on: overlap with the label, median {np.median(carried):.2f}, {100.0 * np.mean(carried >= 0.5):.1f}% at 0.5 or more ({len(carried)} pairs)")
    if not cam_meta and not lidar_meta:
        raise SystemExit("no cam_det.arrows and no pillar_det.arrows in the runs named: nothing to score")


if __name__ == "__main__":
    main(sys.argv[1:])
