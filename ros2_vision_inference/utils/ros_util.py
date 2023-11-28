#!/usr/bin/env python
from typing import Callable, Optional, Union
from typing import Any
import numpy as np
from math import sin, cos
import os
import rospy
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from tf import TransformBroadcaster
import tf
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nuscenes
from .constants import KITTI_NAMES, KITTI_COLORS
from typing import Union, List

class ROSInterface():
    def __init__(self, node_name):
        self.__pub_registry__ = dict()
        self.__sub_registry__ = dict()
        self.__pub_registry__['__marker_array_topics__'] = []
        self.__pub_registry__['__image_topics__'] = []
        self.__pub_registry__['__camera_info_topics__'] = []
        self.__pub_registry__['__image_camera_info_pairs__'] = dict()
        rospy.init_node(node_name)
        rospy.loginfo("Starting RosNode.")
        self.cv_bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster()
    
    def spin(self):
        rospy.spin()
    
    def publish(self, topic, msg):
        self.__pub_registry__[topic].publish(msg)
    
    
    def read_one_parameters(self, param_name, default_value=None):
        return rospy.get_param(f"~{param_name}", default_value)

    def create_publisher(self, msg_type, topic, *args, **kwargs):
        self.__pub_registry__[topic] = rospy.Publisher(topic, msg_type, *args, **kwargs)
        if msg_type == Marker or msg_type == MarkerArray:
            self.__pub_registry__['__marker_array_topics__'].append(topic)
        if msg_type == Image:
            self.__pub_registry__['__image_topics__'].append(topic)
            base_name = os.path.dirname(topic)
            for info_topic in self.__pub_registry__['__camera_info_topics__']:
                if base_name == os.path.dirname(info_topic):
                    self.__pub_registry__['__image_camera_info_pairs__'][topic] = info_topic
                    break
        if msg_type == CameraInfo:
            self.__pub_registry__['__camera_info_topics__'].append(topic)
            base_name = os.path.dirname(topic)
            for image_topic in self.__pub_registry__['__image_topics__']:
                if base_name == os.path.dirname(image_topic):
                    self.__pub_registry__['__image_camera_info_pairs__'][image_topic] = topic
                    break
        return self.__pub_registry__[topic]
    
    def create_timer(self, timer_period_sec: float, callback: Callable, callback_group = None, clock = None):
        self._timer = rospy.Timer(rospy.Duration(timer_period_sec), callback)
        return self._timer

    def create_subscription(self, msg_type, topic: str, callback: Callable, *args, **kwargs):
        sub = rospy.Subscriber(topic, msg_type, callback, *args, **kwargs)
        self.__sub_registry__[topic] = sub
        return sub
    
    def publish_image(self, image, P=None, image_topic=None, camera_info_topic=None, frame_id="base"):
        """Publish image and info message to ROS.

        Args:
            image: numpy.ndArray.
            image_publisher: rospy.Publisher
            camera_info_publisher: rospy.Publisher, should publish CameraInfo
            P: projection matrix [3, 4]. though only [3, 3] is useful.
            frame_id: string, parent frame name.
        """
        image_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding="passthrough")
        image_msg.header.frame_id = frame_id
        image_msg.header.stamp = rospy.Time.now()
        self.__pub_registry__[image_topic].publish(image_msg)

        if P is not None:
            camera_info_msg = CameraInfo()
            camera_info_msg.header.frame_id = frame_id
            camera_info_msg.header.stamp = rospy.Time.now()
            camera_info_msg.height = image.shape[0]
            camera_info_msg.width = image.shape[1]
            camera_info_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info_msg.K = np.reshape(P[0:3, 0:3], (-1)).tolist()
            P_no_translation = np.zeros([3, 4])
            P_no_translation[0:3, 0:3] = P[0:3, 0:3]
            camera_info_msg.p = np.reshape(P_no_translation, (-1)).tolist()

            if camera_info_topic is None:
                if image_topic in self.__pub_registry__['__image_camera_info_pairs__']:
                    camera_info_topic = self.__pub_registry__['__image_camera_info_pairs__'][image_topic]
            self.__pub_registry__[camera_info_topic].publish(camera_info_msg)
    
    def publish_fisheye_image(self, image, fisheye_calib, image_topic, camera_info_topic=None, frame_id='base'):
        """Publish image and info message to ROS. Notice that KITTI360 MEI model is not compatible with OpenCV model, so just a trial.

        Args:
            image: numpy.ndArray.
            image_publisher: rospy.Publisher
            camera_info_publisher: rospy.Publisher, should publish CameraInfo
            P: projection matrix [3, 4]. though only [3, 3] is useful.
            frame_id: string, parent frame name.
        """
        image_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding="passthrough")
        image_msg.header.frame_id = frame_id
        image_msg.header.stamp = rospy.Time.now()
        self.__pub_registry__[image_topic].publish(image_msg)

        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = frame_id
        camera_info_msg.header.stamp = rospy.Time.now()
        camera_info_msg.distortion_model = "equidistant"
        camera_info_msg.height = image.shape[0]
        camera_info_msg.width = image.shape[1]
        k1 = fisheye_calib["distortion_parameters"]['k1']
        k2 = fisheye_calib["distortion_parameters"]['k2']
        camera_info_msg.D = [k1, k2, 0.0, 0.0]
        gamma1 = fisheye_calib["projection_parameters"]['gamma1'] / np.pi
        gamma2 = fisheye_calib["projection_parameters"]['gamma2'] / np.pi
        u0 = fisheye_calib["projection_parameters"]['u0']
        v0 = fisheye_calib["projection_parameters"]['v0']
        K = np.array([[gamma1, 0, u0], [0, gamma2, v0], [0, 0, 1]])
        camera_info_msg.K = np.reshape(K, (-1)).tolist()

        P_no_translation = np.zeros([3, 4])
        P_no_translation[0:3, 0:3] = K[0:3, 0:3]

        camera_info_msg.p = np.reshape(P_no_translation, (-1)).tolist()

        if camera_info_topic is None:
            if image_topic in self.__pub_registry__['__image_camera_info_pairs__']:
                camera_info_topic = self.__pub_registry__['__image_camera_info_pairs__'][image_topic]
        self.__pub_registry__[camera_info_topic].publish(camera_info_msg)

    @staticmethod
    def depth_image_to_point_cloud_array(depth_image, K, parent_frame="left_camera", rgb_image=None, mask=None):
        """  convert depth image into color pointclouds [xyzbgr]
        
        """
        depth_image = np.copy(depth_image)
        w_range = np.arange(0, depth_image.shape[1], dtype=np.float32)
        h_range = np.arange(0, depth_image.shape[0], dtype=np.float32)
        w_grid, h_grid = np.meshgrid(w_range, h_range) #[H, W]
        K_expand = np.eye(4)
        K_expand[0:3, 0:3] = K
        K_inv = np.linalg.inv(K_expand) #[4, 4]

        #[H, W, 4, 1]
        expand_image = np.stack([w_grid * depth_image, h_grid * depth_image, depth_image, np.ones_like(depth_image)], axis=2)[...,np.newaxis]

        pc_3d = np.matmul(K_inv, expand_image)[..., 0:3, 0] #[H, W, 3]
        if rgb_image is not None:
            pc_3d = np.concatenate([pc_3d, rgb_image/256.0], axis=2).astype(np.float32)
        if mask is not None:
            point_cloud = pc_3d[(mask > 0) * (depth_image > 0),:]
        else:
            point_cloud = pc_3d[depth_image > 0,:]
        
        return point_cloud

    @staticmethod
    def line_points_from_3d_bbox_nusc(x, y, z, w, h, l, theta):
        """Compute line points for Rviz Marker from 3D bounding box data in LiDAR coordinate, 

        Args:
            x (float): 
            y (float):
            z (float): 
            w (float): 
            h (float): 
            l (float): 
            theta (float): angular rotation around z axis

        Returns:
            List[Point] : directly usable for Lines Marker
        """    
        corner_matrix = np.array(
            [[-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1]], dtype=np.float32
        )
        relative_eight_corners = 0.5 * corner_matrix * np.array([l, w, h]) #[8, 3]

        _cos = cos(theta)
        _sin = sin(theta)

        rotated_corners_x, rotated_corners_y = (
                relative_eight_corners[:, 0] * _cos +
                    -relative_eight_corners[:, 1] * _sin,
            relative_eight_corners[:, 0] * _sin +
                relative_eight_corners[:, 1] * _cos
            ) #[8]
        rotated_corners = np.stack([rotated_corners_x, rotated_corners_y, relative_eight_corners[:,2]], axis=-1) #[8, 3]
        abs_corners = rotated_corners + np.array([x, y, z])  # [8, 3]

        points = []
        for i in range(1, 5):
            points += [
                Point(x=abs_corners[i, 0], y=abs_corners[i, 1], z=abs_corners[i, 2]),
                Point(x=abs_corners[i%4+1, 0], y=abs_corners[i%4+1, 1], z=abs_corners[i%4+1, 2])
            ]
            points += [
                Point(x=abs_corners[(i + 4)%8, 0], y=abs_corners[(i + 4)%8, 1], z=abs_corners[(i + 4)%8, 2]),
                Point(x=abs_corners[((i)%4 + 5)%8, 0], y=abs_corners[((i)%4 + 5)%8, 1], z=abs_corners[((i)%4 + 5)%8, 2])
            ]
        points += [
            Point(x=abs_corners[2, 0], y=abs_corners[2, 1], z=abs_corners[2, 2]),
            Point(x=abs_corners[7, 0], y=abs_corners[7, 1], z=abs_corners[7, 2]),
            Point(x=abs_corners[3, 0], y=abs_corners[3, 1], z=abs_corners[3, 2]),
            Point(x=abs_corners[6, 0], y=abs_corners[6, 1], z=abs_corners[6, 2]),

            Point(x=abs_corners[4, 0], y=abs_corners[4, 1], z=abs_corners[4, 2]),
            Point(x=abs_corners[5, 0], y=abs_corners[5, 1], z=abs_corners[5, 2]),
            Point(x=abs_corners[0, 0], y=abs_corners[0, 1], z=abs_corners[0, 2]),
            Point(x=abs_corners[1, 0], y=abs_corners[1, 1], z=abs_corners[1, 2])
        ]

        return points

    @staticmethod
    def line_points_from_3d_bbox(x, y, z, w, h, l, theta):
        """
            Compute line points for Rviz Marker from 3D bounding box data in camera coordinate, 

        Args:
            x (float): 
            y (float):
            z (float): 
            w (float): 
            h (float): 
            l (float): 
            theta (float): angular rotation around y axis

        Returns:
            List[Point] : directly usable for Lines Marker
        """
        corner_matrix = np.array(
            [[-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1]], dtype=np.float32
        )
        relative_eight_corners = 0.5 * corner_matrix * np.array([w, h, l]) #[8, 3]

        _cos = cos(theta)
        _sin = sin(theta)

        rotated_corners_x, rotated_corners_z = (
                relative_eight_corners[:, 2] * _cos +
                    relative_eight_corners[:, 0] * _sin,
            -relative_eight_corners[:, 2] * _sin +
                relative_eight_corners[:, 0] * _cos
            ) #[8]
        rotated_corners = np.stack([rotated_corners_x, relative_eight_corners[:,1], rotated_corners_z], axis=-1) #[8, 3]
        abs_corners = rotated_corners + np.array([x, y, z])  # [8, 3]

        points = []
        for i in range(1, 5):
            points += [
                Point(x=abs_corners[i, 0], y=abs_corners[i, 1], z=abs_corners[i, 2]),
                Point(x=abs_corners[i%4+1, 0], y=abs_corners[i%4+1, 1], z=abs_corners[i%4+1, 2])
            ]
            points += [
                Point(x=abs_corners[(i + 4)%8, 0], y=abs_corners[(i + 4)%8, 1], z=abs_corners[(i + 4)%8, 2]),
                Point(x=abs_corners[((i)%4 + 5)%8, 0], y=abs_corners[((i)%4 + 5)%8, 1], z=abs_corners[((i)%4 + 5)%8, 2])
            ]
        points += [
            Point(x=abs_corners[2, 0], y=abs_corners[2, 1], z=abs_corners[2, 2]),
            Point(x=abs_corners[7, 0], y=abs_corners[7, 1], z=abs_corners[7, 2]),
            Point(x=abs_corners[3, 0], y=abs_corners[3, 1], z=abs_corners[3, 2]),
            Point(x=abs_corners[6, 0], y=abs_corners[6, 1], z=abs_corners[6, 2]),

            Point(x=abs_corners[4, 0], y=abs_corners[4, 1], z=abs_corners[4, 2]),
            Point(x=abs_corners[5, 0], y=abs_corners[5, 1], z=abs_corners[5, 2]),
            Point(x=abs_corners[0, 0], y=abs_corners[0, 1], z=abs_corners[0, 2]),
            Point(x=abs_corners[1, 0], y=abs_corners[1, 1], z=abs_corners[1, 2])
        ]

        return points

    def object_to_marker(self, obj, frame_id="base", marker_id=None, color=None, duration=0.1):
        """ Transform an object to a marker.

        Args:
            obj: Dict or Nuscene.dataclasses.Box
            frame_id: string; parent frame name
            marker_id: visualization_msgs.msg.Marker.id
            duration: the existence time in rviz
        
        Return:
            marker: visualization_msgs.msg.Marker

        object dictionary expectation:
            object['whl'] = [w, h, l]
            object['xyz'] = [x, y, z] # center point location in center camera coordinate
            object['theta']: float
            object['score']: float
            object['type_name']: string # should have name in constant.KITTI_NAMES

        """
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        if marker_id is not None:
            marker.id = marker_id
        marker.type = 5
        marker.scale.x = 0.2

        if color is None:
            object_cls_index = KITTI_NAMES.index(obj["type_name"])
            obj_color = KITTI_COLORS[object_cls_index] #[r, g, b]
        else:
            obj_color = color
        marker.color.r = obj_color[0] / 255.0
        marker.color.g = obj_color[1] / 255.0
        marker.color.b = obj_color[2] / 255.0
        if isinstance(obj, dict):
            marker.color.a = obj["score"]
            marker.points = self.line_points_from_3d_bbox(obj["xyz"][0], obj["xyz"][1], obj["xyz"][2], obj["whl"][0], obj["whl"][1], obj["whl"][2], obj["theta"])
        if isinstance(obj, nuscenes.utils.data_classes.Box):
            marker.color.a = 1.0 if np.isnan(obj.score) else float(obj.score)
            marker.points = self.line_points_from_3d_bbox_nusc(
                                obj.center[0], obj.center[1], obj.center[2],
                                obj.wlh[0]   , obj.wlh[2]   , obj.wlh[1]   , obj.orientation.yaw_pitch_roll[0])
        marker.lifetime = rospy.Duration.from_sec(duration)
        return marker

    def publish_transformation(self, T, frame_id, child_frame_id, stamp=None):
        """Publish a transform from frame_id to child_frame_id

        Args:
            T (np.ndarray): array of [3, 4] or [4, 4]
            frame_id (str): base_frame
            child_frame_id (str): child_frame
        """
        homo_R = np.eye(4)
        homo_R[0:3, 0:3] = T[0:3, 0:3]
        self.tf_broadcaster.sendTransform(
            (T[0, 3], T[1, 3], T[2, 3]),
            tf.transformations.quaternion_from_matrix(homo_R),
            rospy.Time.now(),
            child_frame_id,
            frame_id
        )

    def publish_transformation_quat(self, trans, quat, frame_id, child_frame_id, stamp=None):
        """Publish a transform from frame_id to child_frame_id

        Args:
            trans: [xyz]
            quat: [xyzw]
            frame_id (str): base_frame
            child_frame_id (str): child_frame
        """
        msg = TransformStamped()
        if stamp is None:
            stamp = rospy.Time.now()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.child_frame_id = child_frame_id
        msg.transform.translation.x = trans[0]
        msg.transform.translation.y = trans[1]
        msg.transform.translation.z = trans[2]

        msg.transform.rotation.x = quat[0]
        msg.transform.rotation.y = quat[1]
        msg.transform.rotation.z = quat[2]
        msg.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(msg)

    def array2pc2(self, points, parent_frame, field_names='xyza'):
        """ Creates a point cloud message.
        Args:
            points: Nxk array of xyz positions (m) and rgba colors (0..1)
            parent_frame: frame in which the point cloud is defined
            field_names : name for the k channels repectively i.e. "xyz" / "xyza"
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate(field_names)]

        header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

        return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * len(field_names)),
            row_step=(itemsize * len(field_names) * points.shape[0]),
            data=data
        )

    def publish_point_cloud(self, pointcloud, point_cloud_topic, frame_id, field_names='xyza'):
        """Convert point cloud array to PointCloud2 message and publish
        
        Args:
            pointcloud: point cloud array [N,3]/[N,4]
            pc_publisher: ROS publisher for PointCloud2
            frame_id: parent_frame name.
            field_names: name for each channel, ['xyz', 'xyza'...]
        """
        msg = self.array2pc2(pointcloud, frame_id, field_names)
        self.__pub_registry__[point_cloud_topic].publish(msg)

    def clear_all_bbox(self, marker_topic=None):
        clear_marker = Marker()
        clear_marker.action = 3
        if marker_topic is None:
            for marker_topic in self.__pub_registry__['__marker_array_topics__']:
                marker_publisher = self.__pub_registry__[marker_topic]
                array = MarkerArray()
                array.markers = [clear_marker]
                marker_publisher.publish(array)
        else:
            marker_publisher = self.__pub_registry__[marker_topic]
            array = MarkerArray()
            array.markers = [clear_marker]
            marker_publisher.publish(array)

    def clear_bboxes(self, marker_publisher:Union[str, rospy.Publisher], marker_ids:List[int]):
        if isinstance(marker_publisher, str):
            marker_publisher = self.__pub_registry__[marker_publisher]
        assert isinstance(marker_publisher, rospy.Publisher)
        marker_array = MarkerArray()
        for marker_id in marker_ids:
            marker = Marker()
            marker.action = 2 # DELETE=2
            marker.id = marker_id
            marker_array.markers.append(marker)
        marker_publisher.publish(marker_array)
