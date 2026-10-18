# SceneGraphFusion-ext

This repository contains the C++ implementation of modifined version of **SceneGraphFusion**, an online 3D scene graph generation framework that combines RGB-D dense SLAM with YOLO instance segmentation and graph neural network-based graph prediction.


## Prerequisites

### System
```bash
sudo apt update
sudo apt install git build-essential cmake
# or via pip
pip3 install cmake
```

### Eigen3 and OpenCV
```bash
# Eigen 3.3.90+
# OpenCV 4.0.0+
# If not found, CMake will attempt to install them locally.
sudo apt install libeigen3-dev
sudo apt install libopencv-dev
```

### ROS2

### Assimp (mesh loading for rendered view generation)
```bash
sudo apt install libassimp-dev
```

### GUI (optional)
```bash
sudo apt install freeglut3-dev libglfw3-dev
```

### ONNX Runtime (required for graph prediction)

Use **ONNX Runtime 1.16+** (the Tele GNN pipeline runs dynamic-shape graphs —
node/edge counts differ every frame — and ONNX Runtime v1.8.2's
`ScatterElements` CPU kernel has a validation bug that rejects the normal
"more edges than nodes" case, e.g. `Indices dim=20 ... is greater than input
dim=5`, crashing as soon as 3+ nodes are batched into one GNN call. Do not
build v1.8.2 from source per older instructions.

Easiest: download a prebuilt release (CPU or GPU build matching your CUDA
version) from https://github.com/microsoft/onnxruntime/releases and extract
it anywhere, e.g. `~/lib/onnxruntime-linux-x64-gpu-<version>/`. Point the
build at it with `-DONNXROOT_DIR=<extracted_path>` (see Build below); it
does not need to be installed system-wide.

Alternatively, build from source:
```bash
git clone --recursive -b v1.20.1 https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config RelWithDebInfo --build_shared_lib --parallel
cd build/Linux/RelWithDebInfo
sudo make install
```

---

## Build

```bash
git clone <this_repo>
cd SceneGraphFusion
git submodule init
git submodule update
mkdir build && cd build
# Basic build (no GUI, no graph prediction)
cmake ..
# With GUI and graph prediction
cmake -DBUILD_GRAPHPRED=ON -DBUILD_GUI=ON ..
# If ONNX Runtime isn't installed system-wide (e.g. you used a prebuilt release
# instead of `sudo make install`), point CMake at it explicitly:
# cmake -DBUILD_GRAPHPRED=ON -DBUILD_GUI=ON -DONNXROOT_DIR=<extracted_onnxruntime_path> ..
# Also publish the scene graph as a ROS2 topic (see "Publishing to ROS2" below):
# source the ROS2 (and workspace overlay, for scene_msgs) setup.bash first, then
# cmake -DBUILD_GRAPHPRED=ON -DBUILD_GUI=ON -DBUILD_ROS_PUBLISH=ON ..
make -j$(nproc)
```

Binaries are written to `bin/`:
- `bin/exe_GraphSLAM` — non-GUI mode
- `bin/exe_GraphSLAM_GUI` — interactive GUI mode

---

## Dataset structure


```
<sequence>/
  ├── color/          # RGB frames (*.jpg or *.png)
  ├── depth/          # Depth frames (*.png, 16-bit, millimetres)
  └── poses/          # Camera poses (4×4 matrices, one per frame)
```


---

## Run

### Non-GUI (batch processing)

```bash
./bin/exe_GraphSLAM \
    --pth_in    <path/to/sequence> \
    --pth_model <path/to/model_dir> \
    --scene_graph_json_dir <output/dir>
```

### GUI (interactive)

```bash
./bin/exe_GraphSLAM_GUI \
    --pth_in    <path/to/sequence> \
    --pth_model <path/to/model_dir> \
    --scene_graph_json_dir <output/dir>
```

```
./bin/exe_GraphSLAM_GUI   --pth_in /media/gao/gao_ssd/teledataset_/30_2  --pth_model /home/gao/codes/TeleScenegraphFusion-visual-feature/out/telescenegraphfusion_sgfn_yolo_label_relation   --scene_graph_json_dir "/media/gao/gao_ssd/3D scene graph/tmp"
```


### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pth_in` | — | Path to input sequence (required) |
| `--pth_model` | — | Path to directory containing ONNX model files |
| `--scene_graph_json_dir` | — | Output directory for per-frame scene graph JSONs |
| `--scene_graph_ros_topic` | — | ROS2 topic name to publish per-frame scene graph *changes* on (`scene_msgs/TeleScene`); requires `-DBUILD_ROS_PUBLISH=ON` |
| `--tele_yolo_model` | — | Path to YOLOv8 ONNX model for instance segmentation /home/xxx/codes/TeleScenegraphFusion-visual-feature/out/yolo/yolo11n_seg_teledataset-3/weights|
| `--tele_yolo_conf_threshold` | 0.35 | YOLO detection confidence threshold |
| `--tele_yolo_iou_threshold` | 0.70 | NMS IoU threshold |
| `--tele_yolo_object_graph` | false | Enable YOLO-based object graph |
| `--tele_yolo_track_assoc_distance` | 0.60 m | Max distance for track–observation association |
| `--tele_yolo_track_max_age` | -1 | Frames before a track is dropped (-1 = never) |
| `--tele_yolo_object_min_points` | 50 | Min surfel count to accept a detection |
| `--tele_yolo_object_max_points_per_mask` | 2000 | Max surfels sampled per YOLO mask |
| `--tele_yolo_track_max_points` | 2000 | Max accumulated surfels per track |
| `--tele_yolo_object_voxel_size` | 0.030 m | Voxel size for track surfel deduplication |

---

## Publishing to ROS2

Built with `-DBUILD_ROS_PUBLISH=ON` (requires a sourced ROS2 environment, e.g.
`source /opt/ros/<distro>/setup.bash` plus the overlay that provides the
`scene_msgs` package, before running `cmake`), `--scene_graph_ros_topic <name>`
publishes the scene graph as a `scene_msgs/msg/TeleScene` topic instead of (or
alongside) writing JSON files.

Each published message is a **delta**: it only contains nodes/edges that were
added, changed, or removed since the previous frame (`TeleObject.ope_id` /
`Edge.old_relation`+`new_relation`), so message size tracks how much the scene
actually changed rather than the total node/edge count. Nothing is published
on frames with no change. This can be combined with `--scene_graph_json_dir`
to get both outputs at once.

```bash
./bin/exe_GraphSLAM_GUI \
    --pth_in    <path/to/sequence> \
    --pth_model <path/to/model_dir> \
    --scene_graph_ros_topic telescene
```
```
./bin/exe_GraphSLAM_GUI --pth_in /media/gao/gao_ssd/teledataset_/35_3   --pth_model /home/gao/codes/TeleScenegraphFusion-visual-feature/out/telescenegraphfusion_sgfn_yolo_label_relation   --scene_graph_ros_topic telescene
```

---

## Output Format

Each processed frame produces a JSON file `<output_dir>/frame_XXXXXX.json`:

```json
{
  "frame": 1785,
  "camera_pose": [
    [r00, r01, r02, tx],
    [r10, r11, r12, ty],
    [r20, r21, r22, tz],
    [0,   0,   0,   1 ]
  ],
  "nodes": [
    {
      "id": 100003,
      "label": "table",
      "confidence": 0.9627,
      "center": [-1.702, 0.476, 3.340],
      "bbox_max": [-1.202, 0.931, 3.741],
      "bbox_min": [-2.203, 0.020, 2.938]
    }
  ],
  "edges": [
    {
      "from": 100003,
      "to": 100004,
      "relation": "on",
      "confidence": 0.872
    }
  ]
}
```

- `camera_pose`: 4×4 camera-to-world matrix; rotation is unitless, translation is in **metres**
- `center`, `bbox_max`, `bbox_min`: 3D position in **metres** (world frame)
- `confidence`: latest detection/prediction score for the assigned label

---

## GUI Controls

When running in GUI mode:

| Control | Action |
|---------|--------|
| `Space` | Start / pause processing |
| Mouse drag | Rotate view |
| Scroll | Zoom |
| **BBox checkbox** | Toggle 3D bounding box and orientation axis overlay |

The orange arrow on each node indicates the estimated primary orientation axis, computed by 2D PCA on the horizontal (XZ) plane of the object's surfel cloud.

---


## License

See [LICENSE](LICENSE).
