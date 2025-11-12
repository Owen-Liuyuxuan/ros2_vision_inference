import onnxruntime as ort
import numpy as np
import rclpy
import cv2
import os
from .utils.ros_util import ROSInterface
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, CompressedImage
from visualization_msgs.msg import MarkerArray
import threading
from numba import jit
from .seg_labels import PALETTE
import time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from typing import List

MONO3D_NAMES = ['car', 'truck', 'bus', 
                'trailer', 'construction_vehicle',
                'pedestrian', 'motorcycle', 'bicycle',
                'traffic_cone', 'barrier']

COLOR_MAPPINGS = {
    'car' : (  0,  0,142),  'truck': (  0,  0, 70) ,
    'bus': (  0, 60,100), 'trailer': (  0,  0,110),
    'construction_vehicle':  (  0,  0, 70), 'pedestrian': (220, 20, 60),
    'motorcycle': (  0,  0,230), 'bicycle': (119, 11, 32),
    'traffic_cone': (180,165,180), 'barrier': (190,153,153)
}

@jit(nopython=True, cache=True)
def ColorizeSeg(pred_seg, rgb_image, opacity=1.0, palette=PALETTE):
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    h, w = pred_seg.shape
    for i in range(h):
        for j in range(w):
            color_seg[i, j] = palette[pred_seg[i, j]]
    new_image = rgb_image * (1 - opacity) + color_seg * opacity
    new_image = new_image.astype(np.uint8)
    return new_image

def normalize_image(image):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std  = np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - rgb_mean
    image = image / rgb_std
    return image


class BaseInferenceThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build_model(*args, **kwargs)
    def build_model(self, onnx_path, gpu_index=0):
        providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '0', 'device_id': str(gpu_index)})]
        sess_options = ort.SessionOptions()
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers, sess_options=sess_options)
        input_shape = self.ort_session.get_inputs()[0].shape # [B, 3, h, w]
        self.inference_h = input_shape[2]
        self.inference_w = input_shape[3]
    def set_inputs(self, images:List[np.ndarray], P:List[np.ndarray], transforms:List[np.ndarray], masks:List[np.ndarray]):
        self.images = images
        self.P = P
        self.transforms = transforms
        self.masks = masks
    def resize(self, images, Ps, masks):
        return_images = []
        return_Ps     = []
        return_masks  = []
        for image, P, mask in zip(images, Ps, masks):
            h0, w0 = image.shape[0:2]
            scale = min(self.inference_h / h0, self.inference_w / w0)
            h_eff = int(h0 * scale)
            w_eff = int(w0 * scale)
            final_image = np.zeros([self.inference_h, self.inference_w, 3])
            final_image[0:h_eff, 0:w_eff] = cv2.resize(image, (w_eff, h_eff),
                                                        interpolation=cv2.INTER_LINEAR)
            final_mask = np.zeros([self.inference_h, self.inference_w])
            final_mask[0:h_eff, 0:w_eff] = cv2.resize(mask, (w_eff, h_eff),
                                                        interpolation=cv2.INTER_NEAREST)
            P = P.copy()
            P[0:2, :] = P[0:2, :] * scale
            return_images.append(final_image)
            return_Ps.append(P)
            return_masks.append(final_mask)
        return_images = np.stack(return_images, axis=0) # [N, H, W, 3]
        return_Ps     = np.stack(return_Ps, axis=0) # [N, 3, 4]
        return_masks  = np.stack(return_masks, axis=0) # [N, H, W]
        return return_images, return_Ps, return_masks
    def deresize(self, seg):
        seg = seg[0:self.h_eff, 0:self.w_eff]
        seg = cv2.resize(seg, (self.w0, self.h0), interpolation=cv2.INTER_NEAREST)
        return seg
    def run(self):
        raise NotImplementedError
    def join(self):
        threading.Thread.join(self)
        threading.Thread.__init__(self)
        return self._output

class MonodepthThread(BaseInferenceThread):
    def run(self):
        start_time = time.time()
        resized_image, resized_P, masks = self.resize(self.images, self.P, self.masks)
        input_numpy = np.ascontiguousarray(np.transpose(resized_image, (0, 3, 1, 2)), dtype=np.float32)
        P_numpy = np.array(resized_P, dtype=np.float32)
        T = np.array(self.transforms, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        print(f"input_numpy.shape={input_numpy.shape}, P_numpy.shape={P_numpy.shape}, T.shape={T.shape}, masks.shape={masks.shape}")
        outputs = self.ort_session.run(None, {'image': input_numpy, 'P2': P_numpy, 'T': T, 'masks': masks})
        print(outputs[0].shape)
        self._output = outputs[0]
        print(f"monodepth runtime: {time.time() - start_time}")


class VisionInferenceNode():
    def __init__(self):
        self.ros_interface = ROSInterface("VisionInferenceNode")

        self.logger = self.ros_interface.get_logger()
        self.clock = self.ros_interface.get_clock()
        self._read_params()
        self._init_model()
        self._init_static_memory()
        self._init_topics()

        self.logger.info("Initialization Done")
        self.ros_interface.spin()

    def _read_params(self):
        self.logger.info("Reading parameters...")

        self.monodepth_flag = self.ros_interface.read_one_parameters("MONODEPTH_FLAG", True)
        self.mask_base_path = self.ros_interface.read_one_parameters("MASK_BASE_PATH", 
                                                                    "/media/ukenryu/external_ssd/odaiba_dataset")
        
        if self.monodepth_flag:
            self.monodepth_weight_path = self.ros_interface.read_one_parameters("MONODEPTH_CKPT_FILE",
                                        "/home/ukenryu/python_try_new/unsupervised_monodepth/visionfactory/demos/merge_multicam_monodepth_smaller.onnx")
            self.monodepth_gpu_index = int(self.ros_interface.read_one_parameters("MONODEPTH_GPU_INDEX", 0))
        
        self.gpu_index = int(self.ros_interface.read_one_parameters("GPU", 0))
        self.seg_opacity = float(self.ros_interface.read_one_parameters("opacity", 0.9))

    def _init_model(self):
        self.logger.info("Initializing model...")
        if self.monodepth_flag:
            self.monodepth_thread = MonodepthThread(self.monodepth_weight_path, gpu_index=self.monodepth_gpu_index)
        self.logger.info("Model Done")
    
    def _init_static_memory(self):
        self.logger.info("Initializing static memory...")
        # Initialize arrays to track which cameras have been initialized
        self.camera_info_received = [False] * 6
        self.transform_received = [False] * 6
        
        # Initialize data structures for camera info
        self.frame_ids = [None] * 6
        self.P_matrices = [None] * 6
        self.transforms = [None] * 6
        self.num_objects = 0
        
        # Load mask images
        self.masks = []
        for i in range(6):
            mask_path = os.path.join(self.mask_base_path, f"camera{i}", "mask.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Convert to binary mask (0-1)
                mask = (mask > 0).astype(np.float32)
                self.logger.info(f"Loaded mask from {mask_path}, shape: {mask.shape}")
            else:
                self.logger.warn(f"Mask file not found at {mask_path}, using empty mask")
                # Create an empty mask (all ones) if file doesn't exist
                mask = np.ones((720, 1280), dtype=np.float32)  # Default size, adjust if needed
            self.masks.append(mask)
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.ros_interface)

    def _init_topics(self):
        self.ros_interface.create_publisher(PointCloud2, "point_cloud", 10)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)
        
        # Create subscribers for camera images
        self.ros_interface.create_async_subscribers(
            [CompressedImage for _ in range(6)],
            [f'/sensing/camera/camera{i}/image_rect_color/compressed' for i in range(6)],
            self.compressed_images_callback, 
            qos_profile=qos_profile, 
            slop=0.1
        )
        
        # Create subscribers for camera info
        for i in range(6):
            # Create individual subscribers for camera info to get initial transforms
            self.ros_interface.create_subscription(
                CameraInfo,
                f'/sensing/camera/camera{i}/camera_info',
                lambda msg, idx=i: self.single_camera_info_callback(msg, idx),
                qos_profile
            )

    def get_transform_matrix(self, source_frame, target_frame="base_link"):
        """Get the transform matrix from source_frame to target_frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Extract translation
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Extract rotation as quaternion [x, y, z, w]
            quaternion = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            # Convert quaternion to rotation matrix
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = translation
            
            return transform_matrix
        except Exception as e:
            self.logger.error(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return np.eye(4)  # Return identity matrix on failure

    def single_camera_info_callback(self, msg, camera_idx):
        """Callback for individual camera info messages to initialize transforms."""
        if camera_idx >= 6:  # Safety check
            return
            
        # Extract camera matrix
        P_matrix = np.zeros((3, 4))
        P_matrix[0:3, 0:3] = np.array(msg.k).reshape((3, 3))
        self.P_matrices[camera_idx] = P_matrix
        
        # Store frame ID
        self.frame_ids[camera_idx] = msg.header.frame_id
        self.camera_info_received[camera_idx] = True
        
        # Get transform from camera frame to base_link
        if self.frame_ids[camera_idx]:
            try:
                transform = self.get_transform_matrix(self.frame_ids[camera_idx], "base_link")
                self.transforms[camera_idx] = transform
                self.transform_received[camera_idx] = True
                self.logger.info(f"Got transform for camera {camera_idx}: {self.frame_ids[camera_idx]} -> base_link")
            except Exception as e:
                self.logger.error(f"Failed to get transform for camera {camera_idx}: {e}")

    def compressed_images_callback(self, *compressed_msgs):
        """Callback for compressed image messages - receives multiple CompressedImage messages."""
        # Check if we have all camera matrices and transforms
        if not all(self.camera_info_received) or not all(self.transform_received):
            missing_cameras = [i for i, received in enumerate(self.camera_info_received) if not received]
            missing_transforms = [i for i, received in enumerate(self.transform_received) if not received]
            
            if missing_cameras:
                self.logger.info(f"Waiting for camera info for cameras: {missing_cameras}", throttle_duration_sec=1.0)
            if missing_transforms:
                self.logger.info(f"Waiting for transforms for cameras: {missing_transforms}", throttle_duration_sec=1.0)
            return
        
        # Convert compressed images to OpenCV format
        images = []
        for i, msg in enumerate(compressed_msgs):
            if i >= 6:  # Safety check
                continue
            # Convert to BGR format (OpenCV default)
            image = self.ros_interface.cv_bridge.compressed_imgmsg_to_cv2(msg)
            # Convert BGR to RGB
            image = image[..., ::-1]
            images.append(image)
        
        # Process the images
        self.process_images(images)

    def process_images(self, images: List[np.ndarray]):
        """Process multiple camera images."""
        starting = time.time()
        
        if self.monodepth_flag:
            # Set inputs for the monodepth model
            self.monodepth_thread.set_inputs(
                images, 
                self.P_matrices, 
                self.transforms,
                self.masks
            )
            self.monodepth_thread.start()
            
            # Get depth output
            depth = self.monodepth_thread.join()
            depth = depth[::4]
            z = depth[:, 2]
            mask = np.logical_and(-1.0 < z, z < 5.0)
            depth = depth[mask, :]

            # Publish point cloud
            if depth is not None:
                # Assuming depth[1] contains the point cloud data
                print(depth.shape)
                self.ros_interface.publish_point_cloud(
                    depth, 
                    "point_cloud", 
                    frame_id="base_link",  # Use base_link as the frame for the merged point cloud
                    field_names='xyzrgb'
                )

        self.logger.info(f"Total runtime: {time.time() - starting}")

def main(args=None):
    rclpy.init(args=args)
    VisionInferenceNode()

if __name__ == "__main__":
    main()