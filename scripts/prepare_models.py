#!/usr/bin/env python3
"""The detector files `pipes` loads, derived from published ONNX files: the
camera detectors (`pipes run --detector-model`, Megvii's YOLOX) and the lidar
detector (`pipes run --lidar-detector on`, PointPillars as NVIDIA exports it).

    python3 scripts/prepare_models.py            # every model
    python3 scripts/prepare_models.py yolox_s-1024x320
    python3 scripts/prepare_models.py yolox_x-1248x384-fp16
    python3 scripts/prepare_models.py pointpillars-kitti
    python3 scripts/prepare_models.py pointpillars-kitti-fp16

Camera. Each published YOLOX file declares a fixed 1x3x416x416 input. `tract`,
the default runtime, ignores the declared size and runs the graph at the size
the pipeline feeds it; ONNX Runtime does not, and refuses a 1x3x192x640
tensor. So each model `pipes` can run is a derived file: the published graph,
unchanged, with its declared input set to the pipeline's size and its declared
output to the anchor count that size gives. Nothing else is touched, and both
runtimes load the derived file.

Half precision. The two biggest camera models, YOLOX-L and YOLOX-X at
1248x384 (the frame's own size, to the stride), are for the GPU, and a GPU
runs them about half as fast again with their weights as 16-bit floats. Those
derived files are the same graph with every 32-bit float tensor and weight
converted (`onnxconverter-common`), the input and output left as 32-bit so
nothing outside the file changes. On the four drives they find what the
32-bit files find, object for object. The lidar detector has a 16-bit file
too, derived the same way from its 32-bit one, which is the file the GPU
loads: the network runs in two thirds of the time, which is what pays for
running it a second time on the half of the sweep behind the car.

Lidar. NVIDIA's CUDA-PointPillars publishes OpenPCDet's PointPillars (KITTI:
car, pedestrian, cyclist) as an ONNX file built for TensorRT. One node of it,
`PPScatterPlugin`, is a TensorRT plugin no other runtime has: it places each
pillar's 64 features at the pillar's cell of a 496x432 bird's-eye image. The
derived file replaces that one node with standard ONNX that does the same
thing (a zero image, `ScatterND` by cell index, a reshape and a transpose),
lets the number of pillars vary instead of padding every sweep to a fixed
count, and repairs one attribute the exporter wrote as a list where the
standard wants a number (`ReduceMax.keepdims`). No weight changes. Its inputs
are `voxels` [pillars, 32, 10] and `pillar_index` [pillars, 1] (the cell,
row * 432 + column), which `crates/pipes-kitti/src/pillars.rs` builds.

Each published file is downloaded into models/ when it is not there (and
checked against its pinned sha256 either way), the derived file is written
beside it, and its sha256 is printed. Those hashes are pinned in
crates/pipes-kitti/src/camdet.rs and crates/pipes-kitti/src/pillars.rs: a
change here that changes the bytes is a different model, and `pipes` refuses
it at load.

Needs `onnx` (pip install onnx==1.17.0), which brings numpy, and for the
half-precision files `onnxconverter-common` (pip install
onnxconverter-common==1.14.0 numpy==2.1.3: that converter does not run on a
newer numpy). The versions matter beyond that: they decide the bytes written,
and so the hashes `pipes` checks. With `onnxruntime` installed too, each
derived file is run once on zeros to check that it accepts its new shape.
"""
import hashlib
import os
import sys
import urllib.request

import onnx

YOLOX_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
# CUDA-PointPillars at the commit that last changed the model (2023-12-07),
# so the file cannot move under the pin. The model is stored with Git LFS,
# which this address serves.
POINTPILLARS_COMMIT = "490a8fb745f1f5575910e55eb9e31a7818387a74"
POINTPILLARS_URL = (
    "https://media.githubusercontent.com/media/NVIDIA-AI-IOT/CUDA-PointPillars/"
    + POINTPILLARS_COMMIT
    + "/model/pointpillar.onnx"
)
MODELS_DIR = "models"

# The published files: where each comes from, and its sha256.
SOURCES = {
    "yolox_nano.onnx": (
        YOLOX_RELEASE + "yolox_nano.onnx",
        "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    ),
    "yolox_s.onnx": (
        YOLOX_RELEASE + "yolox_s.onnx",
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    ),
    "yolox_m.onnx": (
        YOLOX_RELEASE + "yolox_m.onnx",
        "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f",
    ),
    "yolox_l.onnx": (
        YOLOX_RELEASE + "yolox_l.onnx",
        "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
    ),
    "yolox_x.onnx": (
        YOLOX_RELEASE + "yolox_x.onnx",
        "c892d7aaf1c4746d8a4d675bec669a4db4f434b4ee1efb654bc9b353379c7c55",
    ),
    "pointpillar.onnx": (
        POINTPILLARS_URL,
        "b48eada794191301b1255a327542b772152fd9a9364b2c7c184e11e58bcbf517",
    ),
}

# Camera: name -> (published file, input width, input height, half precision)
DERIVED = {
    "yolox_nano-640x192": ("yolox_nano.onnx", 640, 192, False),
    "yolox_s-640x192": ("yolox_s.onnx", 640, 192, False),
    "yolox_s-1024x320": ("yolox_s.onnx", 1024, 320, False),
    "yolox_m-1024x320": ("yolox_m.onnx", 1024, 320, False),
    "yolox_l-1248x384-fp16": ("yolox_l.onnx", 1248, 384, True),
    "yolox_x-1248x384-fp16": ("yolox_x.onnx", 1248, 384, True),
}

# Lidar: name -> (published file, half precision)
LIDAR_DERIVED = {
    "pointpillars-kitti": ("pointpillar.onnx", False),
    "pointpillars-kitti-fp16": ("pointpillar.onnx", True),
}

# PointPillars' bird's-eye grid, rows (y) by columns (x), and the width of a
# pillar's feature: the model's own.
GRID_Y, GRID_X, PILLAR_FEATURES = 496, 432, 64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name):
    url, want = SOURCES[name]
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    got = sha256(path)
    if want is None:
        print(f"{name}: sha256 {got} ({os.path.getsize(path):,} bytes) -- UNPINNED, pin it")
    elif got != want:
        os.remove(path)
        raise SystemExit(f"{name}: sha256 {got}, expected {want}; deleted")
    else:
        print(f"{name}: sha256 {got} ({os.path.getsize(path):,} bytes)")
    return path


def derive(name, source, w, h, half):
    m = onnx.load(source)
    inp = m.graph.input[0].type.tensor_type.shape.dim
    out = m.graph.output[0].type.tensor_type.shape.dim
    assert len(inp) == 4 and len(out) == 3, (len(inp), len(out))
    inp[0].dim_value, inp[2].dim_value, inp[3].dim_value = 1, h, w
    anchors = (w // 8) * (h // 8) + (w // 16) * (h // 16) + (w // 32) * (h // 32)
    out[0].dim_value, out[1].dim_value, out[2].dim_value = 1, anchors, 85
    # Shapes inferred for the published size would contradict the new one.
    del m.graph.value_info[:]
    if half:
        import warnings

        from onnxconverter_common import float16

        # The converter warns once for every weight too small for 16 bits,
        # which it rounds to the smallest that fits: thousands of lines.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.checker.check_model(m)
    dst = os.path.join(MODELS_DIR, name + ".onnx")
    onnx.save(m, dst)
    words = ", weights as 16-bit floats" if half else ""
    print(f"{name}.onnx: sha256 {sha256(dst)} ({os.path.getsize(dst):,} bytes), input 1x3x{h}x{w}, output 1x{anchors}x85{words}")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return
    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    y = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 3, h, w), np.float32)})[0]
    assert y.shape == (1, anchors, 85), y.shape
    print(f"  runs at that size: output {y.shape}")


def derive_lidar(name, source, half):
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(source)
    g = m.graph
    # The published inputs: voxels [10000, 32, 10], voxel_idxs [10000, 4],
    # voxel_num [1]. The first stays, with a free pillar count; the other two
    # fed the plugin and go with it.
    assert [i.name for i in g.input] == ["voxels", "voxel_idxs", "voxel_num"], [i.name for i in g.input]
    dims = g.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [10000, 32, 10], [d.dim_value for d in dims]
    dims[0].dim_param = "pillars"
    while len(g.input) > 1:
        g.input.pop()
    g.input.append(helper.make_tensor_value_info("pillar_index", TensorProto.INT64, ["pillars", 1]))

    # The two reshapes round the pillar encoder were written for 10000
    # pillars; -1 lets the count vary.
    reshaped = 0
    for init in g.initializer:
        if init.name == "reshape_0_shape":
            init.CopyFrom(numpy_helper.from_array(np.array([-1, 10], np.int64), init.name))
            reshaped += 1
        if init.name == "reshape_1_shape":
            init.CopyFrom(numpy_helper.from_array(np.array([-1, 32, PILLAR_FEATURES], np.int64), init.name))
            reshaped += 1
    assert reshaped == 2, reshaped

    # The exporter wrote `keepdims` as the list [0]; the standard, the checker
    # and ONNX Runtime want the number 0. The same value.
    fixed = 0
    for n in g.node:
        if n.op_type == "ReduceMax":
            kept = [a for a in n.attribute if a.name != "keepdims"]
            del n.attribute[:]
            n.attribute.extend(kept)
            n.attribute.append(helper.make_attribute("keepdims", 0))
            fixed += 1
    assert fixed == 1, fixed

    # The plugin: features [pillars, 64] to a [1, 64, 496, 432] image, each
    # pillar's row at its cell and zeros elsewhere.
    [at] = [k for k, n in enumerate(g.node) if n.op_type == "PPScatterPlugin"]
    plugin = g.node[at]
    features, image = plugin.input[0], plugin.output[0]
    g.initializer.append(numpy_helper.from_array(np.array([-1, PILLAR_FEATURES], np.int64), "pp_feat_shape"))
    g.initializer.append(numpy_helper.from_array(np.array([GRID_Y * GRID_X, PILLAR_FEATURES], np.int64), "pp_dense_shape"))
    g.initializer.append(numpy_helper.from_array(np.array([1, GRID_Y, GRID_X, PILLAR_FEATURES], np.int64), "pp_bev_shape"))
    standard = [
        helper.make_node("Reshape", [features, "pp_feat_shape"], ["pp_feat"], name="pp_reshape_feat"),
        helper.make_node(
            "ConstantOfShape",
            ["pp_dense_shape"],
            ["pp_zeros"],
            name="pp_zeros",
            value=helper.make_tensor("z", TensorProto.FLOAT, [1], [0.0]),
        ),
        helper.make_node("ScatterND", ["pp_zeros", "pillar_index", "pp_feat"], ["pp_scattered"], name="pp_scatter"),
        helper.make_node("Reshape", ["pp_scattered", "pp_bev_shape"], ["pp_bev"], name="pp_reshape_bev"),
        helper.make_node("Transpose", ["pp_bev"], [image], name="pp_transpose", perm=[0, 3, 1, 2]),
    ]
    del g.node[at]
    for k, n in enumerate(standard):
        g.node.insert(at + k, n)
    # Shapes inferred for 10000 pillars would contradict a free count.
    del g.value_info[:]
    m.producer_name = "pipes-rs prepare_models.py"
    if half:
        import warnings

        from onnxconverter_common import float16

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.checker.check_model(m)
    dst = os.path.join(MODELS_DIR, name + ".onnx")
    onnx.save(m, dst)
    words = ", weights as 16-bit floats" if half else ""
    print(f"{name}.onnx: sha256 {sha256(dst)} ({os.path.getsize(dst):,} bytes), inputs voxels [pillars, 32, 10] and pillar_index [pillars, 1]{words}")
    try:
        import onnxruntime as ort
    except ImportError:
        return
    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    feeds = {"voxels": np.zeros((3, 32, 10), np.float32), "pillar_index": np.array([[0], [7], [GRID_Y * GRID_X - 1]], np.int64)}
    shapes = [y.shape for y in s.run(None, feeds)]
    assert shapes == [(1, 248, 216, 18), (1, 248, 216, 42), (1, 248, 216, 12)], shapes
    print(f"  runs on three pillars: outputs {shapes}")


def main(only):
    os.makedirs(MODELS_DIR, exist_ok=True)
    unknown = [n for n in only if n not in DERIVED and n not in LIDAR_DERIVED]
    if unknown:
        raise SystemExit(f"not a model this script derives: {', '.join(unknown)}; it derives {', '.join(list(DERIVED) + list(LIDAR_DERIVED))}")
    cameras = {n: DERIVED[n] for n in (only or DERIVED) if n in DERIVED}
    lidars = {n: LIDAR_DERIVED[n] for n in (only or LIDAR_DERIVED) if n in LIDAR_DERIVED}
    for src in sorted({v[0] for v in cameras.values()} | {v[0] for v in lidars.values()}):
        fetch(src)
    for name, (src, w, h, half) in cameras.items():
        derive(name, os.path.join(MODELS_DIR, src), w, h, half)
    for name, (src, half) in lidars.items():
        derive_lidar(name, os.path.join(MODELS_DIR, src), half)


if __name__ == "__main__":
    main(sys.argv[1:])
