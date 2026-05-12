import json
import struct

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import cv2
import numpy as np
import open3d as o3d

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header, String, Bool
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from scipy.spatial.transform import Rotation


COLOR_RANGES = {
    'red':          [(( 42, 149, 130), (136, 195, 173))],
    'orange':       [(( 87, 146, 156), (202, 183, 183))],
    'yellow':       [((122, 113, 151), (207, 146, 191))],
    'light_green':  [(( 97,  85, 128), (161, 110, 166))],
    'blue':         [(( 88, 106,  80), (172, 126, 103))],
    'mint':         [((111,  87, 117), (228, 114, 134))],
    'white':        [((167, 113,  93), (209, 140, 118))],
    'purple':       [(( 89, 129,  77), (169, 151, 116))],
    'brown':        [(( 52, 126, 124), (154, 155, 146))],
    'pink':         [((117, 159, 119), (184, 189, 152))],
    'light_blue':   [((132, 105,  85), (255, 140, 130))],
}

STUD_PITCH_M           = 0.016
MAX_BRICK_FOOTPRINT_M2 = (2 * STUD_PITCH_M) * (6 * STUD_PITCH_M) * 2.0
MIN_BRICK_FOOTPRINT_M2 = (1 * STUD_PITCH_M) * (2 * STUD_PITCH_M) * 0.4

BRICK_HEIGHTS = {
    'half':   0.0096,
    'normal': 0.0192,
    'tall':   0.0384,
}

HEIGHT_THRESHOLDS = {
    'half':   (0.000, 0.013),
    'normal': (0.013, 0.030),
    'tall':   (0.030, 9.999),
}

HEIGHT_PERCENTILE    = 90
MIN_CLUSTER_PTS      = 10

TABLE_MASK_MARGIN_M  = 0.018
MAX_BRICK_HEIGHT_M   = 0.060

VOXEL_SIZE_M         = 0.004
DBSCAN_EPS_M         = 0.012
DBSCAN_MIN_PTS       = 10

COLOR_MATCH_MIN_FRAC = 0.40
TF_TIMEOUT_SEC       = 1.0

BRICK_DEBUG_COLORS = {
    'red':         (220,  50,  50),
    'orange':      (230, 130,  30),
    'yellow':      (230, 210,  30),
    'light_green': (100, 200,  80),
    'blue':        ( 50,  80, 200),
    'mint':        ( 80, 200, 180),
    'white':       (230, 230, 230),
    'purple':      (130,  50, 200),
    'brown':       (140,  80,  40),
    'pink':        (220, 130, 180),
    'light_blue':  (100, 180, 230),
}


class BrickDetectorNode(Node):

    def __init__(self):
        super().__init__('brick_detector')

        self.latest_xyz    = None
        self.latest_pc_bgr = None
        self._enabled      = False

        self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points',
            self.pointcloud_callback, 10)
        self.create_subscription(
            Bool, '/brick_detection_enabled',
            lambda msg: setattr(self, '_enabled', msg.data), 10)

        self.pose_pub  = self.create_publisher(PoseArray,    '/detected_bricks',       10)
        self.meta_pub  = self.create_publisher(String,       '/detected_bricks_meta',  10)
        self.debug_pub = self.create_publisher(PointCloud2,  '/detected_bricks_debug', 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')

        self.create_timer(3.0, self.process)
        self.get_logger().info('BrickDetectorNode initialized.')

    def pointcloud_callback(self, msg: PointCloud2):
        self.latest_xyz, self.latest_pc_bgr = self._unpack_pointcloud(msg)

    def _get_cam_to_base(self) -> np.ndarray | None:
        """Return 4x4 camera→base_link transform matrix, or None if unavailable."""
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                'baseplate_frame', camera_frame,
                rclpy.time.Time(), timeout=Duration(seconds=TF_TIMEOUT_SEC))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}', once=True)
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = [t.x, t.y, t.z]
        return T

    def _transform_pts_to_base(self, pts: np.ndarray, cam_to_base: np.ndarray) -> np.ndarray:
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        return (cam_to_base @ pts_h.T).T[:, :3]

    def process(self):
        if not self._enabled:
            return
        if self.latest_xyz is None or self.latest_pc_bgr is None:
            return

        cam_to_base = self._get_cam_to_base()
        if cam_to_base is None:
            return

        clusters      = self._cluster_above_table(self.latest_xyz, self.latest_pc_bgr.copy(), cam_to_base)
        all_bricks    = []
        debug_clusters = []

        for cluster_pts_base, cluster_bgr in clusters:
            color_name = self._classify_cluster_color(cluster_bgr)
            if color_name is None:
                continue

            if self._cluster_footprint_m2(cluster_pts_base) < MIN_BRICK_FOOTPRINT_M2:
                continue

            rows, cols, brick_R = self._shape_from_pointcloud_extent(cluster_pts_base)

            centroid  = np.median(cluster_pts_base, axis=0)
            heights_z = cluster_pts_base[:, 2]
            heights_pos = heights_z[heights_z > 0]
            height_m  = (float(np.percentile(heights_pos, HEIGHT_PERCENTILE))
                         if len(heights_pos) >= MIN_CLUSTER_PTS // 2
                         else BRICK_HEIGHTS['normal'])

            pose = Pose()
            pose.position.x = float(centroid[0])
            pose.position.y = float(centroid[1])
            pose.position.z = float(centroid[2])
            q = Rotation.from_matrix(brick_R).as_quat()
            pose.orientation.x = float(q[0])
            pose.orientation.y = float(q[1])
            pose.orientation.z = float(q[2])
            pose.orientation.w = float(q[3])

            all_bricks.append({
                'color':       color_name,
                'shape':       (rows, cols),
                'height_type': self._classify_height(height_m),
                'height_m':    height_m,
                'pose':        pose,
            })
            debug_clusters.append((cluster_pts_base, BRICK_DEBUG_COLORS.get(color_name, (180, 180, 180))))

        self.publish_poses(all_bricks)
        self._publish_debug_cloud(debug_clusters)

    def _unpack_pointcloud(self, msg: PointCloud2):
        H, W     = msg.height, msg.width
        n_floats = msg.point_step // 4
        fields   = {f.name: f.offset // 4 for f in msg.fields}
        data     = np.frombuffer(msg.data, dtype=np.float32).reshape(H * W, n_floats)

        xyz = np.stack([
            data[:, fields['x']].reshape(H, W),
            data[:, fields['y']].reshape(H, W),
            data[:, fields['z']].reshape(H, W),
        ], axis=2)
        xyz[~np.isfinite(xyz)] = np.nan

        rgb_u32 = data[:, fields['rgb']].view(np.uint32)
        bgr = np.stack([
            (rgb_u32         & 0xFF).astype(np.uint8).reshape(H, W),
            ((rgb_u32 >>  8) & 0xFF).astype(np.uint8).reshape(H, W),
            ((rgb_u32 >> 16) & 0xFF).astype(np.uint8).reshape(H, W),
        ], axis=2)
        return xyz, bgr

    def _cluster_above_table(self, xyz_cam: np.ndarray, bgr: np.ndarray,
                             cam_to_base: np.ndarray) -> list:
        pts_cam = xyz_cam.reshape(-1, 3)
        colors  = bgr.reshape(-1, 3)

        valid     = np.all(np.isfinite(pts_cam), axis=1)
        pts_base  = self._transform_pts_to_base(pts_cam[valid], cam_to_base)
        col_valid = colors[valid]

        z    = pts_base[:, 2]
        keep = (z > TABLE_MASK_MARGIN_M) & (z < MAX_BRICK_HEIGHT_M)

        if np.sum(keep) < DBSCAN_MIN_PTS:
            return []

        above_pts = pts_base[keep]
        above_col = col_valid[keep]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(above_pts)
        pcd.colors = o3d.utility.Vector3dVector(above_col.astype(np.float64) / 255.0)

        pcd_down    = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_M)
        pts_down    = np.asarray(pcd_down.points)
        colors_down = (np.asarray(pcd_down.colors) * 255).astype(np.uint8)

        labels = np.array(pcd_down.cluster_dbscan(
            eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_PTS, print_progress=False))

        clusters = []
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            clusters.append((pts_down[mask], colors_down[mask]))
        return clusters

    def _cluster_footprint_m2(self, pts_base: np.ndarray) -> float:
        if len(pts_base) < 3:
            return 0.0
        proj = pts_base[:, :2].astype(np.float32)
        hull = cv2.convexHull(proj.reshape(-1, 1, 2))
        return float(cv2.contourArea(hull))

    def _classify_cluster_color(self, cluster_bgr: np.ndarray):
        lab_pts = cv2.cvtColor(
            cluster_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)

        l_vals  = lab_pts[:, 0]
        lab_pts = lab_pts[(l_vals > 30) & (l_vals < 240)]
        if len(lab_pts) < 5:
            return None

        def match_mask(pts, ranges):
            mask = np.zeros(len(pts), dtype=bool)
            for lower, upper in ranges:
                mask |= np.all(
                    (pts >= np.array(lower, np.float32)) &
                    (pts <= np.array(upper, np.float32)), axis=1)
            return mask

        orange_mask = match_mask(lab_pts, COLOR_RANGES['orange'])
        pink_mask   = match_mask(lab_pts, COLOR_RANGES['pink'])

        best_color, best_count = None, 0
        for color_name, ranges in COLOR_RANGES.items():
            m = match_mask(lab_pts, ranges)
            if color_name == 'red':
                m = m & ~orange_mask & ~pink_mask
            count = int(np.sum(m))
            if count > best_count:
                best_count = count
                best_color = color_name

        if best_count < len(lab_pts) * COLOR_MATCH_MIN_FRAC:
            return None
        return best_color

    def _shape_from_pointcloud_extent(self, pts_base: np.ndarray) -> tuple:
        if len(pts_base) < MIN_CLUSTER_PTS:
            return 1, 1, np.eye(3)

        xy       = pts_base[:, :2]
        centered = xy - xy.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_pca = centered @ Vt.T

        long_m  = float(np.percentile(proj_pca[:, 0], 95) - np.percentile(proj_pca[:, 0], 5))
        short_m = float(np.percentile(proj_pca[:, 1], 95) - np.percentile(proj_pca[:, 1], 5))

        cols = max(1, round(long_m  / STUD_PITCH_M))
        rows = max(1, round(short_m / STUD_PITCH_M))

        x_ax = np.array([Vt[0, 0], Vt[0, 1], 0.0])
        x_ax /= np.linalg.norm(x_ax)
        z_ax  = np.array([0.0, 0.0, 1.0])
        y_ax  = np.cross(z_ax, x_ax); y_ax /= np.linalg.norm(y_ax)
        x_ax  = np.cross(y_ax, z_ax); x_ax /= np.linalg.norm(x_ax)
        R_mat = np.column_stack([x_ax, y_ax, z_ax])
        return rows, cols, R_mat

    def _classify_height(self, height_m: float) -> str:
        for label, (lo, hi) in HEIGHT_THRESHOLDS.items():
            if lo <= height_m < hi:
                return label
        return 'normal'

    def _publish_debug_cloud(self, debug_clusters: list):
        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        rows = []
        for pts, (r, g, b) in debug_clusters:
            rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
            for pt in pts:
                rows.append([float(pt[0]), float(pt[1]), float(pt[2]), rgb_packed])

        header           = Header()
        header.stamp     = self.get_clock().now().to_msg()
        header.frame_id  = 'baseplate_frame'
        self.debug_pub.publish(pc2.create_cloud(header, fields, rows))

    def publish_poses(self, bricks: list):
        pose_array                 = PoseArray()
        pose_array.header          = Header()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'baseplate_frame'
        meta_list = []

        for brick in bricks:
            pose_array.poses.append(brick['pose'])
            meta_list.append({
                'color':       brick['color'],
                'shape':       list(brick['shape']),
                'height_type': brick['height_type'],
                'height_m':    round(brick['height_m'], 4),
            })

        self.pose_pub.publish(pose_array)
        meta_msg      = String()
        meta_msg.data = json.dumps(meta_list)
        self.meta_pub.publish(meta_msg)
        self.get_logger().info(f'Published {len(pose_array.poses)} bricks.')


def main(args=None):
    rclpy.init(args=args)
    node = BrickDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
