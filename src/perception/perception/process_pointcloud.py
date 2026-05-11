"""
process_pointcloud.py — shoe point cloud extractor.

Flow:
  1. Wait for first synchronized color + aligned depth frame.
  2. Run CV pipeline (GroundingDINO + SAM) on the color image.
  3. Save segments.json + mask .npy files to artifacts_cv/.
  4. Launch web UI (blocking) — user selects/deselects valid segmentations.
  5. Read approved_segments.json, build filtered point cloud, publish.

Publishes (per approved segment i = 0, 1, …):
  /shoe_points_{i}     (PointCloud2, base_link) — top-slice filtered points for segment i
  /shoe_goal_point_{i} (PointStamped, base_link) — median point for segment i
  /filtered_points     (PointCloud2, base_link) — full scene for RViz
"""

import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, '/home/cc/ee106a/sp26/class/ee106a-acz/Desktop/eecs106a-FINAL/cv')

import cv2
import message_filters
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

_LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

from pipeline import CVPipeline
from save_segments import save_segments
from segment_image import visualise

ARTIFACTS_DIR = Path('/home/cc/ee106a/sp26/class/ee106a-acz/Desktop/eecs106a-FINAL (segmentation pipeline)/artifacts_cv')
WEB_UI_DIR    = Path('/home/cc/ee106a/sp26/class/ee106a-acz/Desktop/eecs106a-FINAL (segmentation pipeline)/web_ui')

Z_SLICE_M = 0.0008   # thickness of top-of-shoe slice to keep (metres)

_CANONICAL_TYPES = ['sneaker', 'flip flop', 'slipper', 'shoe']

def _canonical_footwear_type(label: str) -> str | None:
    """Return the canonical footwear type for a label, or None if not footwear.
    Handles descriptive labels like 'white sneakers' or 'brown flip flop'.
    """
    label = label.lower()
    for t in _CANONICAL_TYPES:
        if t in label:
            return t
    return None


class ShoePointCloudExtractor(Node):
    def __init__(self):
        super().__init__('shoe_pointcloud_extractor')

        self.bridge = CvBridge()
        self.camera_info = None
        self.done = False
        self._shoe_pubs: list = []
        self._goal_pubs: list = []
        self._pose_pubs: list = []
        self.goal_msgs:  list = []
        self.pose_msgs:  list = []

        self._depth_m: np.ndarray | None = None
        self._img_stamp = None

        print('[init] Loading CV pipeline (GroundingDINO + SAM) — this takes ~30-60s...')
        self.cv_pipeline = CVPipeline()
        print('[init] CV pipeline ready.')

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self._camera_info_cb,
            1,
        )

        print('[init] Subscribing to color image + aligned depth...')
        img_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw'
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, depth_sub], queue_size=5, slop=0.1
        )
        self.sync.registerCallback(self._synced_cb)

        self.full_pub = self.create_publisher(PointCloud2, '/filtered_points', 10)
        self.create_timer(0.5, self._republish_goal)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        print('[init] Ready. Waiting for camera data...')

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            K = msg.k
            print(f'[init] Camera intrinsics received: fx={K[0]:.1f} fy={K[4]:.1f} cx={K[2]:.1f} cy={K[5]:.1f}')

    def _unproject(self, depth_m: np.ndarray, fx: float, fy: float,
                   cx: float, cy: float) -> np.ndarray:
        """Convert aligned depth image (H, W) in metres to (H, W, 3) XYZ in camera frame."""
        H, W = depth_m.shape
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        X = (uu - cx) * depth_m / fx
        Y = (vv - cy) * depth_m / fy
        return np.stack([X, Y, depth_m], axis=-1)  # (H, W, 3)

    def _synced_cb(self, img_msg: Image, depth_msg: Image):
        if self.done:
            return
        if self.camera_info is None:
            print('Waiting for camera info...')
            return

        self.done = True  # prevent re-entry from subsequent frames
        # Offload to a background thread so rclpy.spin() keeps running
        # (keeps TF buffer alive for the lookup after the web UI confirms)
        threading.Thread(
            target=self._run_pipeline,
            args=(img_msg, depth_msg),
            daemon=True,
        ).start()

    def _run_pipeline(self, img_msg: Image, depth_msg: Image):
        print('[1/5] Got synchronized color + aligned depth frame.')

        # --- Capture frame ---
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
        pil_img = PILImage.fromarray(cv_img)
        H, W = cv_img.shape[:2]
        print(f'      Image size: {W}x{H}')

        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self._depth_m = depth_raw.astype(np.float32) / 1000.0
        self._depth_m[self._depth_m == 0] = np.nan
        self._img_stamp = img_msg.header.stamp

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        pil_img.save(ARTIFACTS_DIR / '01_raw_image.png')
        print(f'      Saved {ARTIFACTS_DIR / "01_raw_image.png"}')

        # --- Run CV pipeline ---
        print('[2/5] Running CV pipeline (GroundingDINO + SAM)...')
        detections = self.cv_pipeline.run(pil_img, min_mask_coverage=0.0, max_mask_coverage=0.20)
        print(f'      Detections: {[(d.label, round(d.confidence, 2)) for d in detections]}')
        shoes = [d for d in detections if _canonical_footwear_type(d.label) is not None]

        if not shoes:
            print('      No footwear detected — retrying next frame.')
            self.done = False
            return

        print(f'      Found {len(shoes)} footwear instance(s).')

        # --- Save segments for web UI ---
        print('[3/5] Saving segments...')
        save_segments(pil_img, shoes, out_dir=ARTIFACTS_DIR)
        visualise(pil_img, shoes).save(ARTIFACTS_DIR / '02_segmentation_overlay.png')
        print(f'      Written segments.json + masks to {ARTIFACTS_DIR}')

        # --- Launch web UI and block this background thread until user confirms ---
        print('[4/5] Launching web UI — open http://localhost:5000 and confirm your selection.')
        _spec = importlib.util.spec_from_file_location('web_ui_server', WEB_UI_DIR / 'server.py')
        web_server = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(web_server)

        approval_event = threading.Event()
        ui_thread = threading.Thread(
            target=web_server.run_approval_server,
            args=(ARTIFACTS_DIR, approval_event),
            daemon=True,
        )
        ui_thread.start()
        approval_event.wait()   # blocks background thread; rclpy.spin() keeps running

        # --- Publish approved point clouds ---
        print('[5/5] User confirmed. Building approved point clouds...')
        self._publish_approved_clouds()
        print('[5/5] _publish_approved_clouds() returned.')

    def _publish_approved_clouds(self):
        print('      Reading approved_segments.json...')
        approved_path = ARTIFACTS_DIR / 'approved_segments.json'
        segments_path = ARTIFACTS_DIR / 'segments.json'

        if not approved_path.exists():
            print('      approved_segments.json not found — aborting.')
            return

        approved_json = json.loads(approved_path.read_text())
        approved_ids = set(approved_json.get('approved_ids', []))
        label_overrides = {int(k): v for k, v in approved_json.get('label_overrides', {}).items()}
        seg_data = json.loads(segments_path.read_text())
        img_w, img_h = seg_data['image_size']  # [width, height]

        K = self.camera_info.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        cloud_xyz = self._unproject(self._depth_m, fx, fy, cx, cy)  # compute once

        print('      Looking up TF transform...')
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_color_optical_frame', rclpy.time.Time()
            )
        except Exception as e:
            print(f'      TF lookup failed: {e}')
            return
        print('      TF transform OK.')

        q = tf.transform.rotation
        t = tf.transform.translation
        rot  = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        tvec = np.array([t.x, t.y, t.z])

        base_header = Header()
        base_header.frame_id = 'base_link'
        base_header.stamp = self._img_stamp

        pub_idx = 0
        for seg in seg_data['segments']:
            if seg['id'] not in approved_ids:
                continue

            mask_path = ARTIFACTS_DIR / f"mask_{seg['id']}.npy"
            if not mask_path.exists():
                print(f'      Warning: mask_{seg["id"]}.npy missing, skipping.')
                continue

            mask = np.load(mask_path)
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)

            pts = cloud_xyz[mask]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.size == 0:
                print(f'      Segment {seg["id"]}: no valid depth — skipping.')
                continue

            pts_full = pts.copy()  # unfiltered, camera frame
            min_z = pts[:, 2].min()
            pts = pts[pts[:, 2] <= min_z + Z_SLICE_M]

            pts_base = (rot @ pts.T).T + tvec
            pts_full_base = (rot @ pts_full.T).T + tvec

            # Gripper orientation from direction_uv
            dx, dy = seg.get('direction_uv', [1.0, 0.0])
            perp_cam = np.array([dx, dy, 0.0])
            perp_base = rot @ perp_cam
            gripper_yaw = float(np.arctan2(perp_base[1], perp_base[0]))
            gq = (Rotation.from_euler('z', gripper_yaw) * Rotation.from_quat([0, 1, 0, 0])).as_quat()

            effective_label = _canonical_footwear_type(
                label_overrides.get(seg['id'], seg['label'])
            ) or seg['label']

            if effective_label == 'sneaker':
                # Split lengthwise along long axis
                long_axis_base = rot @ np.array([dx, dy, 0.0])
                norm = np.linalg.norm(long_axis_base)
                if norm > 0:
                    long_axis_base /= norm
                centroid = np.mean(pts_base, axis=0)
                proj = (pts_base - centroid) @ long_axis_base

                mask_a = proj >= 0
                mask_b = ~mask_a
                half_a, half_b = pts_base[mask_a], pts_base[mask_b]

                depth_a = pts[mask_a, 2].mean() if mask_a.any() else float('inf')
                depth_b = pts[mask_b, 2].mean() if mask_b.any() else float('inf')
                if depth_a <= depth_b:
                    top_pts, bot_pts = half_a, half_b
                    bot_proj_mask_full = (pts_full_base - centroid) @ long_axis_base < 0
                else:
                    top_pts, bot_pts = half_b, half_a
                    bot_proj_mask_full = (pts_full_base - centroid) @ long_axis_base >= 0

                bot_full_pts = pts_full_base[bot_proj_mask_full]

                # Top and bottom half clouds get _top/_bot suffixes; pub_idx is shared
                if len(top_pts) > 0:
                    top_pub = self.create_publisher(PointCloud2, f'/shoe_points_{pub_idx}_top', _LATCHED_QOS)
                    top_pub.publish(pc2.create_cloud_xyz32(base_header, top_pts.tolist()))
                    self._shoe_pubs.append(top_pub)
                if len(bot_pts) > 0:
                    bot_pub = self.create_publisher(PointCloud2, f'/shoe_points_{pub_idx}_bot', _LATCHED_QOS)
                    bot_pub.publish(pc2.create_cloud_xyz32(base_header, bot_pts.tolist()))
                    self._shoe_pubs.append(bot_pub)

                # shoe_goal_point = median of bottom half (heel end)
                goal_pts = bot_pts if len(bot_pts) > 0 else pts_base
                median_base = np.median(goal_pts, axis=0)
                goal = PointStamped()
                goal.header  = base_header
                goal.point.x = float(median_base[0])
                goal.point.y = float(median_base[1])
                goal.point.z = float(median_base[2])
                goal_pub = self.create_publisher(PointStamped, f'/shoe_goal_point_{pub_idx}', _LATCHED_QOS)
                goal_pub.publish(goal)
                self._goal_pubs.append(goal_pub)
                self.goal_msgs.append(goal)

                pose = PoseStamped()
                pose.header      = base_header
                pose.pose.position.x = float(median_base[0])
                pose.pose.position.y = float(median_base[1])
                pose.pose.position.z = float(median_base[2])
                pose.pose.orientation.x = float(gq[0])
                pose.pose.orientation.y = float(gq[1])
                pose.pose.orientation.z = float(gq[2])
                pose.pose.orientation.w = float(gq[3])
                pose_pub = self.create_publisher(PoseStamped, f'/shoe_goal_pose_{pub_idx}', _LATCHED_QOS)
                pose_pub.publish(pose)
                self._pose_pubs.append(pose_pub)
                self.pose_msgs.append(pose)

                # Heel sub-topics from bottom half unfiltered points
                if len(bot_full_pts) > 0:
                    full_center = np.median(bot_full_pts, axis=0)
                    full_goal = PointStamped()
                    full_goal.header = base_header
                    full_goal.point.x = float(full_center[0])
                    full_goal.point.y = float(full_center[1])
                    full_goal.point.z = float(full_center[2])
                    full_pub = self.create_publisher(
                        PointStamped, f'/shoe_goal_point_{pub_idx}_full', _LATCHED_QOS)
                    full_pub.publish(full_goal)
                    self._goal_pubs.append(full_pub)
                    self.goal_msgs.append(full_goal)
                    print(f'        → /shoe_goal_point_{pub_idx}_full (unfiltered bottom center)')

                    bot_proj = (bot_full_pts - centroid) @ long_axis_base
                    thresh = np.percentile(np.abs(bot_proj), 90)
                    heel_pts = bot_full_pts[np.abs(bot_proj) >= thresh]

                    if len(heel_pts) > 0:
                        heel_cloud_pub = self.create_publisher(
                            PointCloud2, f'/shoe_heel_points_{pub_idx}', _LATCHED_QOS)
                        heel_cloud_pub.publish(
                            pc2.create_cloud_xyz32(base_header, heel_pts.tolist()))
                        self._shoe_pubs.append(heel_cloud_pub)

                        max_z_heel = heel_pts[:, 2].max()
                        heel_sliced = heel_pts[heel_pts[:, 2] >= max_z_heel - Z_SLICE_M]
                        if len(heel_sliced) > 0:
                            heel_med = np.median(heel_sliced, axis=0)
                            heel_goal = PointStamped()
                            heel_goal.header = base_header
                            heel_goal.point.x = float(heel_med[0])
                            heel_goal.point.y = float(heel_med[1])
                            heel_goal.point.z = float(heel_med[2])
                            heel_pt_pub = self.create_publisher(
                                PointStamped, f'/shoe_heel_point_{pub_idx}', _LATCHED_QOS)
                            heel_pt_pub.publish(heel_goal)
                            self._goal_pubs.append(heel_pt_pub)
                            self.goal_msgs.append(heel_goal)
                            print(f'        → /shoe_heel_points_{pub_idx} '
                                  f'({len(heel_pts)} pts), /shoe_heel_point_{pub_idx}')

                print(f'      Segment {seg["id"]} (sneaker): '
                      f'{len(top_pts)} top + {len(bot_pts)} bot pts '
                      f'→ /shoe_points_{pub_idx}_top/_bot, goal → /shoe_goal_point_{pub_idx}')

            else:
                # Non-sneaker: single point cloud, no half splitting
                shoe_pub = self.create_publisher(PointCloud2, f'/shoe_points_{pub_idx}', _LATCHED_QOS)
                shoe_pub.publish(pc2.create_cloud_xyz32(base_header, pts_base.tolist()))
                self._shoe_pubs.append(shoe_pub)

                median_base = np.median(pts_base, axis=0)
                goal = PointStamped()
                goal.header  = base_header
                goal.point.x = float(median_base[0])
                goal.point.y = float(median_base[1])
                goal.point.z = float(median_base[2])
                goal_pub = self.create_publisher(PointStamped, f'/shoe_goal_point_{pub_idx}', _LATCHED_QOS)
                goal_pub.publish(goal)
                self._goal_pubs.append(goal_pub)
                self.goal_msgs.append(goal)

                pose = PoseStamped()
                pose.header      = base_header
                pose.pose.position.x = float(median_base[0])
                pose.pose.position.y = float(median_base[1])
                pose.pose.position.z = float(median_base[2])
                pose.pose.orientation.x = float(gq[0])
                pose.pose.orientation.y = float(gq[1])
                pose.pose.orientation.z = float(gq[2])
                pose.pose.orientation.w = float(gq[3])
                pose_pub = self.create_publisher(PoseStamped, f'/shoe_goal_pose_{pub_idx}', _LATCHED_QOS)
                pose_pub.publish(pose)
                self._pose_pubs.append(pose_pub)
                self.pose_msgs.append(pose)

                print(f'      Segment {seg["id"]} ({seg["label"]}): {len(pts_base)} pts '
                      f'→ /shoe_points_{pub_idx}, goal → /shoe_goal_point_{pub_idx}')

            pub_idx += 1

        if pub_idx == 0:
            print('      No approved segments with valid depth — nothing published.')
            return

        all_pts = cloud_xyz.reshape(-1, 3)
        all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
        all_pts_base = (rot @ all_pts.T).T + tvec
        self.full_pub.publish(pc2.create_cloud_xyz32(base_header, all_pts_base.tolist()))
        print(f'      Published {pub_idx} segment(s), {len(all_pts_base)} full-scene pts → /filtered_points')
        print('Done.')

    def _republish_goal(self):
        for goal_msg, goal_pub in zip(self.goal_msgs, self._goal_pubs):
            goal_pub.publish(goal_msg)
        for pose_msg, pose_pub in zip(self.pose_msgs, self._pose_pubs):
            pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ShoePointCloudExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
