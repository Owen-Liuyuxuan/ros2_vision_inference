# ROS2 Vision Inference

This repo contains a unified inference multi-threading nodes for monocular 3D object detection, depth prediction and semantic segmentation.

![RVIZ Example](docs/ros2_vision_inference.gif)

You could checkout the ROS1 version of each inference package:

- [Monodepth ROS1](https://github.com/Owen-Liuyuxuan/monodepth_ros)
- [Mono3D ROS1](https://github.com/Owen-Liuyuxuan/visualDet3D_ros)

In this repo, we fully re-structure the code and messages formats for ROS2 (humble), and integrate multi-thread inferencing for three vision tasks.

- Currently all pretrained models are trained using the [visionfactory](https://github.com/Owen-Liuyuxuan/visionfactory) repo. Thus focusing on out-door autonomous driving scenarios. But it is ok to plugin ONNX models that satisfiy the [interface](#onnx-model-interface).


## Getting Started

This repo relies on [ROS2](https://docs.ros.org/en/humble/Installation.html) and onnxruntime:

```bash
pip3 install onnxruntime-gpu
```

For Orin, find the pip wheel in https://elinux.org/Jetson_Zoo with version number > 1.3 is ok

Under the workspace directory, find the [launch file](./launch/detect_seg_depth.launch.xml) and change topic names and onnx checkpoint paths then build the workspace (you can choose to either launch any of the three tasks in the launch file)

```bash
colcon build --synlink-install
source install/setup.bash
ros2 launch ros2_vision_inference detect_seg_depth.launch.xml
```

Notice that as a known issue from [ros2 python package](https://github.com/ros2/launch/issues/187). The launch/rviz files are copied but not symlinked when we run "colcon build",
so **whenever we modify the launch file, we need to rebuild the package**; whenever we want to modify the rviz file, we need to save it explicitly in the src folder.

```bash
colcon build --symlink-install --packages-select=ros2_vision_inference # rebuilding only ros2_vision_inference
```

## Interface

### Subscribed Topics

/image_raw ([sensor_msgs/Image](https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html))

/camera_info ([sensor_msgs/CameraInfo](https://docs.ros2.org/latest/api/sensor_msgs/msg/CameraInfo.html))

### Publishing Topics

/depth_image ([sensor_msgs/Image](https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html)): Depth image of float type.

/seg_image ([sensor_msgs/Image](https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html)): RGB-colorized segmentation images.

/mono3d/bbox ([visualization_msgs/MarkerArray](https://docs.ros2.org/foxy/api/visualization_msgs/msg/MarkerArray.html)): 3D bounding boxes outputs.

/point_cloud ([sensor_msgs/PointCloud2](https://docs.ros2.org/latest/api/sensor_msgs/msg/PointCloud2.html)): projecting /seg_image with /depth_image into a point cloud.

## ONNX Model Interface

MonoDepth ONNX: `def forward(self, normalized_images[1, 3, H, W], P[1, 3, 4]):->float[1, 1, H, W]`

Segmentation ONNX `def forward(self, normalized_images[1, 3, H, W]):->long[1, H, W]`

Mono3D ONNX: `def forward(self, normalized_images[1, 3, H, W], P[1, 3, 4]):->scores[N], bboxes[N,12], cls_indexes[N]`

Classes definitions are from the [visionfactory](https://github.com/Owen-Liuyuxuan/visionfactory) repo.
