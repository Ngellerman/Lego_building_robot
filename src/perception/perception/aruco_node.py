import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

ARUCO_DICT_ID       = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID     = 0
ARUCO_MARKER_SIZE_M = 0.05
STUD_PITCH_M        = 0.016
BASEPLATE_ROWS      = 16
BASEPLATE_COLS      = 16
N_POSE_SAMPLES      = 30

ARUCO_TO_BP_OFFSET = np.array([
    3.5,
    3.5,
    0.0,
], dtype=np.float64)


class ArucoNode(Node):

    def __init__(self):
        super().__init__('aruco_node')
        self.bridge = CvBridge()

        self.fx = self.fy = self.cx = self.cy = None
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        self._last_tf: TransformStamped = None
        self._pose_samples = []
        self._tf_locked    = False

        self.declare_parameter('camera_frame', 'camera_color_optical_frame')

        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._cam_info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._image_cb, 10)

        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_timer(0.1, self._republish_tf)

        self.get_logger().info('ArucoNode initialized.')

    def _cam_info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]
        if msg.d:
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)

    def _image_cb(self, msg: Image):
        if self.fx is None or self._tf_locked:
            return

        rgb  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        cam_mat = np.array([[self.fx, 0, self.cx],
                             [0, self.fy, self.cy],
                             [0,       0,       1]], dtype=np.float32)

        aruco_dict   = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
        aruco_params = cv2.aruco.DetectorParameters_create()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is None:
            return

        for i, mid in enumerate(ids.flatten()):
            if mid != ARUCO_MARKER_ID:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], ARUCO_MARKER_SIZE_M, cam_mat, self.dist_coeffs)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            R_mat      = cv2.Rodrigues(rvec)[0]
            # TODO: re-enable offset once axis directions are confirmed
            origin_cam = tvec + R_mat @ ARUCO_TO_BP_OFFSET

            camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
            try:
                tf_to_base = self.tf_buffer.lookup_transform(
                    'base_link', camera_frame,
                    rclpy.time.Time(), timeout=Duration(seconds=0.1))
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                self.get_logger().warn('aruco_node: no TF from base_link to camera frame',
                                       throttle_duration_sec=5.0)
                return

            from geometry_msgs.msg import Pose
            import tf2_geometry_msgs
            pose_cam = Pose()
            pose_cam.position.x = float(origin_cam[0])
            pose_cam.position.y = float(origin_cam[1])
            pose_cam.position.z = float(origin_cam[2])
            q_bp = Rotation.from_matrix(R_mat).as_quat()
            pose_cam.orientation.x = float(q_bp[0])
            pose_cam.orientation.y = float(q_bp[1])
            pose_cam.orientation.z = float(q_bp[2])
            pose_cam.orientation.w = float(q_bp[3])

            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, tf_to_base)

            t_sample = np.array([pose_base.position.x,
                                  pose_base.position.y,
                                  pose_base.position.z])
            q_sample = np.array([pose_base.orientation.x, pose_base.orientation.y,
                                  pose_base.orientation.z, pose_base.orientation.w])
            self._pose_samples.append((t_sample, q_sample))

            n = len(self._pose_samples)
            self.get_logger().info(f'Pose sample {n}/{N_POSE_SAMPLES} collected.')

            if n < N_POSE_SAMPLES:
                break

            # Average translations only — orientation is fixed since the baseplate
            # is always flat on the table and aligned with base_link axes.
            translations = np.array([s[0] for s in self._pose_samples])
            mean_t = translations.mean(axis=0)

            t = TransformStamped()
            t.header.stamp            = self.get_clock().now().to_msg()
            t.header.frame_id         = 'base_link'
            t.child_frame_id          = 'baseplate_frame'
            t.transform.translation.x = float(mean_t[0])
            t.transform.translation.y = float(mean_t[1])
            t.transform.translation.z = float(mean_t[2])
            t.transform.rotation.x    = 0.0
            t.transform.rotation.y    = 0.0
            t.transform.rotation.z    = 0.0
            t.transform.rotation.w    = 1.0

            self._last_tf   = t
            self._tf_locked = True
            self.tf_broadcaster.sendTransform(t)
            self.get_logger().info(
                f'baseplate_frame locked from {N_POSE_SAMPLES} averaged samples. '
                f'Position: ({mean_t[0]:.4f}, {mean_t[1]:.4f}, {mean_t[2]:.4f})')
            break

    def _republish_tf(self):
        if self._last_tf is None:
            return
        t = TransformStamped()
        t.header.stamp            = self.get_clock().now().to_msg()
        t.header.frame_id         = self._last_tf.header.frame_id
        t.child_frame_id          = self._last_tf.child_frame_id
        t.transform               = self._last_tf.transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()