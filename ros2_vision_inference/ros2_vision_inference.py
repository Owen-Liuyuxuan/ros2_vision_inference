import onnxruntime as ort
import numpy as np
import rclpy
import cv2
from .utils.ros_util import ROSInterface
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import MarkerArray
import threading
from numba import jit
from .seg_labels import PALETTE
import time

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
    
    def build_model(self, onnx_path):
        self.ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        input_shape = self.ort_session.get_inputs()[0].shape # [1, 3, h, w]
        self.inference_h = input_shape[2]
        self.inference_w = input_shape[3]
    
    def set_inputs(self, image, P=None):
        self.image = image
        self.P = P

    def resize(self, image, P=None):
        self.h0, self.w0 = image.shape[0:2]
        scale = min(self.inference_h / self.h0, self.inference_w / self.w0)
        self.scale = scale
        self.h_eff = int(self.h0 * scale)
        self.w_eff = int(self.w0 * scale)
        final_image = np.zeros([self.inference_h, self.inference_w, 3])
        final_image[0:self.h_eff, 0:self.w_eff] = cv2.resize(image, (self.w_eff, self.h_eff),
                                                    interpolation=cv2.INTER_LINEAR)
        if P is None:
            return final_image
        else:
            P = P.copy()
            P[0:2, :] = P[0:2, :] * scale
            return final_image, P

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

class Mono3D(BaseInferenceThread):
    def run(self):
        global MONO3D_NAMES
        start_time = time.time()
        resized_image, resized_P = self.resize(self.image, self.P)
        input_numpy = np.ascontiguousarray(np.transpose(normalize_image(resized_image), (2, 0, 1))[None], dtype=np.float32)
        P_numpy = np.array(resized_P, dtype=np.float32)[None]
        outputs = self.ort_session.run(None, {'image': input_numpy, 'P2': P_numpy})
        scores = np.array(outputs[0], dtype=np.float) # N
        bboxes = np.array(outputs[1], dtype=np.float) # N, 12
        cls_indexes = outputs[2] # N

        cls_names = [MONO3D_NAMES[cls_index] for cls_index in cls_indexes]

        objects = []
        N = len(bboxes)
        
        
        for i in range(N):
            obj = {}
            obj['whl'] = bboxes[i, 7:10]
            obj['theta'] = bboxes[i, 11]
            obj['score'] = scores[i]
            obj['type_name'] = cls_names[i]
            obj['xyz'] = bboxes[i, 4:7]
            objects.append(obj)
        self._output = objects
        print(f"mono3d runtime: {time.time() - start_time}")

class SegmentationThread(BaseInferenceThread):
    def run(self):
        start_time = time.time()
        resized_image = self.resize(self.image)
        input_numpy = np.ascontiguousarray(np.transpose(normalize_image(resized_image), (2, 0, 1))[None], dtype=np.float32)
        outputs = self.ort_session.run(None, {'input': input_numpy})
        self._output = self.deresize(np.array(outputs[0][0], np.uint8))
        print(f"segmentation runtime: {time.time() - start_time}")

class MonodepthThread(BaseInferenceThread):
    def run(self):
        start_time = time.time()
        resized_image, resized_P = self.resize(self.image, self.P)
        input_numpy = np.ascontiguousarray(np.transpose(normalize_image(resized_image), (2, 0, 1))[None], dtype=np.float32)
        P_numpy = np.array(resized_P, dtype=np.float32)[None]
        outputs = self.ort_session.run(None, {'image': input_numpy, 'P2': P_numpy})
        self._output = self.deresize(outputs[0][0, 0])
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

        self.mono3d_flag = self.ros_interface.read_one_parameters("MONO3D_FLAG", True)
        self.seg_flag = self.ros_interface.read_one_parameters("SEG_FLAG", True)
        self.monodepth_flag = self.ros_interface.read_one_parameters("MONODEPTH_FLAG", True)
        
        
        if self.mono3d_flag:
            self.mono3d_weight_path  = self.ros_interface.read_one_parameters("MONO3D_CKPT_FILE",
                                    "/home/yliuhb/vision_factory/weights/mono3d.onnx")
        
        if self.seg_flag:
            self.seg_weight_path = self.ros_interface.read_one_parameters("SEG_CKPT_FILE",
                                        "/home/yliuhb/vision_factory/weights/seg.onnx")
        
        if self.monodepth_flag:
            self.monodepth_weight_path = self.ros_interface.read_one_parameters("MONODEPTH_CKPT_FILE",
                                        "/home/yliuhb/vision_factory/weights/monodepth.onnx")
        
        self.gpu_index = int(self.ros_interface.read_one_parameters("GPU", 0))
        self.seg_opacity = float(self.ros_interface.read_one_parameters("opacity", 0.9))

    def _init_model(self):
        self.logger.info("Initializing model...")
        if self.mono3d_flag:
            self.mono3d_thread = Mono3D(self.mono3d_weight_path)
        if self.seg_flag:
            self.seg_thread    = SegmentationThread(self.seg_weight_path)
        if self.monodepth_flag:
            self.monodepth_thread = MonodepthThread(self.monodepth_weight_path)
        self.logger.info("Model Done")
    
    def _init_static_memory(self):
        self.logger.info("Initializing static memory...")
        self.frame_id = None
        self.P = None
        self.num_objects = 0
    

    def _init_topics(self):
        self.bbox_publish = self.ros_interface.create_publisher(MarkerArray, "mono3d/bbox", 10)
        self.ros_interface.create_publisher(Image, "seg_image", 10)
        self.ros_interface.create_publisher(Image, "depth_image", 10)
        self.ros_interface.create_publisher(PointCloud2, "point_cloud", 10)

        self.ros_interface.create_subscription(CameraInfo, "/camera_info", self.camera_info_callback, 1)
        self.ros_interface.create_subscription(Image, "/image_raw", self.camera_callback, 1)
        self.ros_interface.clear_all_bbox()


    def camera_info_callback(self, msg:CameraInfo):
        self.P = np.zeros((3, 4))
        self.P[0:3, 0:3] = np.array(msg.k.reshape((3, 3)))
        self.frame_id = msg.header.frame_id

    def camera_callback(self, msg:Image):
        if self.P is None:
            self.logger.info("Waiting for camera info...", throttle_duration_sec=0.5)
            return # wait for camera info
        height = msg.height
        width  = msg.width
        
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))[:, :, ::-1]

        starting = time.time()
        if self.mono3d_flag:
            self.mono3d_thread.set_inputs(image, self.P.copy())
            self.mono3d_thread.start()
        if self.seg_flag:
            self.seg_thread.set_inputs(image)
            self.seg_thread.start()
        if self.monodepth_flag:
            self.monodepth_thread.set_inputs(image, self.P.copy())
            self.monodepth_thread.start()


        objects = self.mono3d_thread.join() if self.mono3d_flag else None
        seg = self.seg_thread.join() if self.seg_flag else None
        depth = self.monodepth_thread.join() if self.monodepth_flag else None

        self.logger.info(f"Total runtime: {time.time() - starting}")

        # publish objects
        if self.mono3d_flag:
            marker_array = MarkerArray()
            for i, obj in enumerate(objects):
                class_name = obj['type_name']
                marker = self.ros_interface.object_to_marker(obj,
                                                            self.frame_id,
                                                            i,
                                                            color=COLOR_MAPPINGS[class_name],
                                                            duration=0.5) # color maps
                marker_array.markers.append(marker)
            self.bbox_publish.publish(marker_array)

        # publish colorized seg
        if self.seg_flag:
            seg_image = ColorizeSeg(seg, image, opacity=self.seg_opacity)
            self.ros_interface.publish_image(seg_image[:, :, ::-1], image_topic="seg_image", frame_id=self.frame_id)

        # publish depth
        if self.monodepth_flag:
            self.ros_interface.publish_image(depth, image_topic="depth_image", frame_id=self.frame_id)

        # publish colorized point cloud
        if self.seg_flag and self.monodepth_flag:
            point_cloud = self.ros_interface.depth_image_to_point_cloud_array(depth, self.P[0:3, 0:3], rgb_image=seg_image)
            mask = (point_cloud[:, 1] > -3.5) * (point_cloud[:, 1] < 5.5) * (point_cloud[:, 2] < 80)
            point_cloud = point_cloud[mask]
            self.ros_interface.publish_point_cloud(point_cloud, "point_cloud", frame_id=self.frame_id, field_names='xyzrgb')

def main(args=None):
    rclpy.init(args=args)
    VisionInferenceNode()

if __name__ == "__main__":
    main()
