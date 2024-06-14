# ROS2 Vision Inference

This repo contains a unified multi-threading inferencing nodes for monocular 3D object detection, depth prediction and semantic segmentation.

![RVIZ Example](docs/ros2_vision_inference.gif)

You could checkout the ROS1 version of each inference package:

- [Monodepth ROS1](https://github.com/Owen-Liuyuxuan/monodepth_ros)
- [Mono3D ROS1](https://github.com/Owen-Liuyuxuan/visualDet3D_ros)

**Update 1**: We have make a ROS1 version of this repo. Please use it with `git checkout ros1`.

**Update 2**: We have added support for [Metric3D](https://github.com/YvanYin/Metric3D/tree/main), which is a state-of-the-art model for depth with scale. It can predict depth with reliable scale while generalize very well on various scenarios. We produce a reliable onnx version of the model, that could runs with TensorRT even on Jetson machines. We also extents its output to point clouds, facilitating robotic applications.

In this repo, we fully re-structure the code and messages formats for ROS2 (humble), and integrate multi-thread inferencing for three vision tasks.

- Currently all pretrained models are trained using the [visionfactory](https://github.com/Owen-Liuyuxuan/visionfactory) repo. Thus focusing on out-door autonomous driving scenarios. But it is ok to plugin ONNX models that satisfiy the [interface](#onnx-model-interface). Published models description:

| Model                          | Type             | Link                                                                                                                  | Description                                                                                                  |
| ------------------------------ | ---------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| monodepth_res101_384_1280.onnx | MonoDepth        | [link](https://github.com/Owen-Liuyuxuan/ros2_vision_inference/releases/download/v1.0/monodepth_res101_384_1280.onnx) | FSNet, res101 backbone, model input shape (384x1280) trained on KITTI/KITTI360/nuscenes                      |
| metric_3d .onnx                | MonoDepth        | [link](https://github.com/Owen-Liuyuxuan/ros2_vision_inference/releases/download/v1.1/metric_3d.onnx)                 | Metric3Dv2, ViT backbone, supervised depth contains full pipeline from depth image to point cloud.           |
| bisenetv1.onnx                 | Segmentation     | [link](https://github.com/Owen-Liuyuxuan/ros2_vision_inference/releases/download/v1.0/bisenetv1.onnx)                 | BiSeNetV1, model input shape (512x768) trained on remapped KITTI360/ApolloScene/CityScapes/BDD100k/a2d2      |
| mono3d_yolox_576_768.onnx      | Mono3D Detection | [link](https://github.com/Owen-Liuyuxuan/ros2_vision_inference/releases/download/v1.0/mono3d_yolox_576_768.onnx)      | YoloX-m MonoFlex, model input (576x768) trained on KITTI/nuscenes/ONCE/bdd100k/cityscapes                    |
| dla34_deform_576_768.onnx      | Mono3D Detection | [link](https://github.com/Owen-Liuyuxuan/ros2_vision_inference/releases/download/v1.1.1/dla34_deform_576_768.onnx)    | DLA34 Deformable Upsample MonoFlex, model input (576x768) trained on KITTI/nuscenes/ONCE/bdd100k/cityscapes  |
| dla34_deform_384_1280.onnx     | Mono3D Detection | [link](https://github.com/Owen-Liuyuxuan/ros2_vision_inference/releases/download/v1.1.1/dla34_deform_384_1280.onnx)   | DLA34 Deformable Upsample MonoFlex, model input (384x1280) trained on KITTI/nuscenes/ONCE/bdd100k/cityscapes |


## Getting Started

This repo relies on [ROS2](https://docs.ros.org/en/humble/Installation.html) and onnxruntime:

> If you want to use ROS1, checkout to ROS1 branch with `git checkout ros1`. We tested the ROS1 code in ROS noetic Ubuntu 20.04 (we need python3 so we suggest we at least run at Ubuntu 20.04). The branch `ros1` is a standard ROS1 package built with `catkin_make` and run with `roslaunch`

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

Metric3D ONNX: `def forward(self, unnormalized_images[1, 3, 616, 1064], P[1, 3, 4], P_inv[1, 4, 4], T[1, 4, 4], mask[1, 616, 1064]):->float[1, 1, H, W], float[HW, 6], bool[HW, 6]`


Classes definitions are from the [visionfactory](https://github.com/Owen-Liuyuxuan/visionfactory) repo.

## Data, Domain

### Reshape Scheme

 We resize the input image to the length/width, and pad zero on the others. We also modify the camera intrinsic accordingly before feeding into the onnx model. The output will be de-resized to the original shape. Currently published models are all trained with various input images, so the node should work naturally with different image sources.

### Expected Data Domain

This is related to the training data of the onnx models. The published models now mainly work on autonomous driving/road scenes. Most of the data for the published segmentation models only use the front-facing camera (Detection / MonoDepth are trained with various camera views). 