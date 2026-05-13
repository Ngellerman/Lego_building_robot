# ROS Libraries
from std_srvs.srv import Trigger
import argparse
import json
import math
from enum import Enum, auto
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
import tf2_ros
import tf2_geometry_msgs
import threading
from planning.ik import IKPlanner


# TODO: replace with real Cartesian poses (x, y, z, qx, qy, qz, qw) in base_link frame
# Default orientation (0, 1, 0, 0) points the gripper straight down.
ARUCO_SCAN_POSE = (-0.4, 0.4, 0.288, 0.0, 1.0, 0.0, 0.0)
BRICK_SCAN_POSE = (0.4, 0.4, 0.288, 0.0, 1.0, 0.0, 0.0)

STUD_PITCH_M        = 0.016   # 16 mm per stud
LEGO_LAYER_HEIGHT_M = 0.0192  # 19.2 mm per normal brick layer

# ── Pick heights (absolute z in base_link) ───────────────────────────────────
PICK_Z_BASE_M       = 0.159   # z of brick top surface in base_link — tune this
PICK_APPROACH_M     = 0.200   # hover height above PICK_Z_BASE_M
PICK_GRASP_M        = 0.000   # grasp height above PICK_Z_BASE_M
PICK_RETRACT_M      = 0.200   # post-pick lift height above PICK_Z_BASE_M

# ── Place heights (absolute z in base_link) ───────────────────────────────────
PLACE_SURFACE_Z_BASE_M = 0.140   # z of layer-1 baseplate surface in base_link — tune this
PLACE_APPROACH_M       = 0.080   # hover height above layer surface
PLACE_DEPOSIT_M        = 0.033   # final descent onto stud
PLACE_RETRACT_M        = 0.150   # post-place lift height above layer surface

_KNOWN_COLORS = {
    'red', 'orange', 'yellow', 'light_green', 'blue',
    'mint', 'white', 'purple', 'brown', 'pink', 'light_blue',
}

BRICK_SCAN_TIMEOUT_S = 15.0  # seconds to wait for a matching brick detection


def _rotation_deg_to_quat(deg):
    """Return (qx, qy, qz, qw) for a down-facing gripper yawed by deg about world Z."""
    r = math.radians(deg)
    # Compose: q_yaw(z) * q_down(y=180°) → (-sin(r/2), cos(r/2), 0, 0)
    return (-math.sin(r / 2), math.cos(r / 2), 0.0, 0.0)

_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
_SAFE_POSITIONS = [
    4.739357948303223,    # shoulder_pan_joint
    -1.841379781762594,   # shoulder_lift_joint
    -1.435956358909607,   # elbow_joint
    -1.3928968769362946,  # wrist_1_joint
    1.593016266822815,    # wrist_2_joint
    -3.146789614354269,   # wrist_3_joint
]

def _make_safe_joint_state():
    js = JointState()
    js.name = _JOINT_NAMES
    js.position = _SAFE_POSITIONS
    return js

SAFE_JOINT_STATE = _make_safe_joint_state()


class Phase(Enum):
    MOVE_TO_ARUCO      = auto()
    WAIT_ARUCO_SCAN    = auto()
    MOVE_TO_BRICK_SCAN = auto()
    WAIT_BRICK_SCAN    = auto()
    PICK_AND_PLACE     = auto()
    DONE               = auto()



def _load_bricks(json_path: Path, picks_path: Path = None) -> list[dict]:
    raw = json.loads(json_path.read_text())

    # lego_build.json format: top-level list of steps with grid positions
    if isinstance(raw, list):
        picks = []
        if picks_path is not None:
            picks = json.loads(picks_path.read_text())['bricks']

        bricks = []
        for i, entry in enumerate(raw):
            qx, qy, qz, qw = _rotation_deg_to_quat(entry.get('rotation_deg', 0))
            pick = picks[i]['pick'] if i < len(picks) else None
            bricks.append({
                'id':    entry.get('block_id', entry.get('step')),
                'label': f"{entry.get('color', '')} {entry.get('type', '')}".strip(),
                'pick':  pick,
                'place': {
                    'x': entry['grid_x'] * -STUD_PITCH_M,
                    'y': entry['grid_z'] * STUD_PITCH_M,
                    'z': 0,
                    'layer': entry.get('layer', 1),
                    'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
                    'frame': 'baseplate',
                },
            })
        return bricks

    # Legacy bricks_test.json format: {"bricks": [...]}
    return raw['bricks']


class LegoBuilder(Node):
    def __init__(self, bricks_json: Path, picks_json: Path = None):
        super().__init__('lego_builder')

        self.bricks    = _load_bricks(bricks_json, picks_json)
        self.brick_idx = 0
        self.phase     = Phase.MOVE_TO_ARUCO

        self.joint_state  = None
        self.job_queue    = []
        self.baseplate_tf = None

        self._latest_poses       = []
        self._latest_detections  = []
        self._detection_lock     = threading.Lock()
        self._detection_event    = threading.Event()
        self._scanning_active    = False

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f'Loaded {len(self.bricks)} brick(s) from {bricks_json}')

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 1)
        self.create_subscription(
            PoseArray, '/detected_bricks', self._detection_pose_cb, 10)
        self.create_subscription(
            String, '/detected_bricks_meta', self._detection_meta_cb, 10)
        self._detection_enable_pub = self.create_publisher(
            Bool, '/brick_detection_enabled', 1)
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.get_logger().info('Loading IK planner...')
        self.ik_planner = IKPlanner()
        self.get_logger().info('IK planner ready.')

        threading.Thread(target=self._wait_and_start, daemon=True).start()

    def _wait_and_start(self):
        import time
        while self.joint_state is None:
            time.sleep(0.1)
        self.get_logger().info('Joint state received — starting state machine.')
        self._advance()

    # ── State machine ───────────────────────────────────────────────────────────

    def _advance(self):
        self.get_logger().info(f'[STATE] {self.phase.name}')
        self.job_queue = []

        if self.phase == Phase.MOVE_TO_ARUCO:
            def _go_aruco():
                js = self.ik_planner.compute_ik(self.joint_state, *ARUCO_SCAN_POSE)
                if js is None:
                    self.get_logger().error('IK failed for ArUco scan pose.')
                    rclpy.shutdown()
                    return
                self.job_queue.append(js)
                self.execute_jobs()
            threading.Thread(target=_go_aruco, daemon=True).start()

        elif self.phase == Phase.WAIT_ARUCO_SCAN:
            def _wait_for_aruco():
                import time
                self.get_logger().info('Waiting for baseplate_frame TF...')
                while rclpy.ok():
                    try:
                        self.baseplate_tf = self.tf_buffer.lookup_transform(
                            'base_link', 'baseplate_frame',
                            rclpy.time.Time())
                        self.get_logger().info('ArUco detected — baseplate_frame acquired.')
                        self.phase = Phase.MOVE_TO_BRICK_SCAN
                        self._advance()
                        return
                    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                        time.sleep(0.2)
            threading.Thread(target=_wait_for_aruco, daemon=True).start()

        elif self.phase == Phase.MOVE_TO_BRICK_SCAN:
            def _go_scan():
                js = self.ik_planner.compute_ik(SAFE_JOINT_STATE, *BRICK_SCAN_POSE)
                if js is None:
                    self.get_logger().error('IK failed for brick scan pose.')
                    rclpy.shutdown()
                    return
                self.job_queue.append(SAFE_JOINT_STATE)
                self.job_queue.append(js)
                self.execute_jobs()
            threading.Thread(target=_go_scan, daemon=True).start()

        elif self.phase == Phase.WAIT_BRICK_SCAN:
            def _wait_for_detection():
                import time
                brick = self.bricks[self.brick_idx]
                label = brick.get('label', '')
                self.get_logger().info(f'Scanning for: "{label}"')
                self._set_scanning(True)
                self._detection_event.clear()

                with self._detection_lock:
                    snapshot = list(self._latest_detections)
                if snapshot:
                    self.get_logger().info(
                        f'Detections at scan pose ({len(snapshot)} brick(s)):')
                    for i, d in enumerate(snapshot):
                        p = d['pose']
                        self.get_logger().info(
                            f'  [{i}] color={d["color"]} shape={d["shape"]} '
                            f'height={d["height_m"]:.4f}m '
                            f'pos=({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f})')
                else:
                    self.get_logger().info('No detections yet at scan pose.')

                deadline = time.time() + BRICK_SCAN_TIMEOUT_S
                while time.time() < deadline:
                    self._detection_event.wait(timeout=1.0)
                    self._detection_event.clear()
                    match = self._match_brick(label)
                    if match is None:
                        continue
                    p = match['pose']
                    # Rotate the unit x-axis by the detected quaternion to get
                    # the brick's long-axis direction, then read off its yaw.
                    ox, oy, oz, ow = (p.orientation.x, p.orientation.y,
                                      p.orientation.z, p.orientation.w)
                    long_x = 1.0 - 2.0 * (oy*oy + oz*oz)
                    long_y = 2.0 * (ox*oy + oz*ow)
                    brick_yaw_deg = math.degrees(math.atan2(long_y, long_x)) + 90.0
                    pqx, pqy, pqz, pqw = _rotation_deg_to_quat(brick_yaw_deg)
                    brick['pick'] = {
                        'x':     p.position.x,
                        'y':     p.position.y,
                        'z':     0,
                        'qx':    pqx,
                        'qy':    pqy,
                        'qz':    pqz,
                        'qw':    pqw,
                        'frame': 'baseplate',
                    }
                    self.get_logger().info(
                        f'Matched "{label}" at baseplate_frame '
                        f'({p.position.x:.3f}, {p.position.y:.3f}, z={match["height_m"]:.3f})')
                    self._set_scanning(False)
                    self.phase = Phase.PICK_AND_PLACE
                    self._advance()
                    return

                self._scanning_active = False
                self.get_logger().error(
                    f'No match for "{label}" after {BRICK_SCAN_TIMEOUT_S:.0f}s — skipping.')
                self.brick_idx += 1
                self.phase = Phase.MOVE_TO_BRICK_SCAN
                self._advance()

            threading.Thread(target=_wait_for_detection, daemon=True).start()

        elif self.phase == Phase.PICK_AND_PLACE:
            if self.brick_idx >= len(self.bricks):
                self.phase = Phase.DONE
                self._advance()
                return
            self._build_pick_place_queue()
            if self.job_queue:
                threading.Thread(target=self.execute_jobs, daemon=True).start()
            else:
                self.get_logger().warn(f'Skipping brick {self.brick_idx} (IK failed).')
                self.brick_idx += 1
                self.phase = Phase.MOVE_TO_BRICK_SCAN
                self._advance()

        elif self.phase == Phase.DONE:
            self.get_logger().info('All bricks placed. Shutting down.')
            rclpy.shutdown()

    def _on_queue_done(self):
        """Called by execute_jobs when the queue empties. Drives phase transitions."""
        if self.phase == Phase.MOVE_TO_ARUCO:
            self.phase = Phase.WAIT_ARUCO_SCAN
            self._advance()

        elif self.phase == Phase.MOVE_TO_BRICK_SCAN:
            self.phase = Phase.WAIT_BRICK_SCAN
            self._advance()

        elif self.phase == Phase.PICK_AND_PLACE:
            self.get_logger().info(f'Brick {self.brick_idx} complete.')
            self.brick_idx += 1
            self.phase = Phase.MOVE_TO_BRICK_SCAN
            self._advance()

    # ── Queue construction ──────────────────────────────────────────────────────

    def _transform_to_base(self, x, y, z, qx, qy, qz, qw):
        """Transform a pose from baseplate_frame into base_link."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        p = tf2_geometry_msgs.do_transform_pose(pose, self.baseplate_tf)
        return p.position.x, p.position.y, p.position.z, \
               p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w

    def _resolve_pose(self, pose_dict):
        """Return (x,y,z,qx,qy,qz,qw) in base_link, transforming if frame=='baseplate'."""
        x  = pose_dict['x'];            y  = pose_dict['y'];  z  = pose_dict['z']
        qx = pose_dict.get('qx', 0.0); qy = pose_dict.get('qy', 1.0)
        qz = pose_dict.get('qz', 0.0); qw = pose_dict.get('qw', 0.0)
        if pose_dict.get('frame') == 'baseplate':
            return self._transform_to_base(x, y, z, qx, qy, qz, qw)
        return x, y, z, qx, qy, qz, qw

    def _build_pick_place_queue(self):
        brick = self.bricks[self.brick_idx]
        pk    = brick['pick']
        pl    = brick['place']
        label = brick.get('label', f'brick_{self.brick_idx}')

        if pk is None:
            self.get_logger().error(f'No pick pose for brick {self.brick_idx} — skipping.')
            return

        px, py, _, pqx, pqy, pqz, pqw = self._resolve_pose(pk)
        pz = PICK_Z_BASE_M

        lx, ly, _, lqx, lqy, lqz, lqw = self._resolve_pose(pl)
        lz = PLACE_SURFACE_Z_BASE_M + (pl.get('layer', 1) - 1) * LEGO_LAYER_HEIGHT_M

        self.get_logger().info(
            f'[Brick {self.brick_idx}] {label} | '
            f'pick=({px:.3f},{py:.3f},{pz:.3f}) place=({lx:.3f},{ly:.3f},{lz:.3f})')

        pre_grasp = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, px, py, pz + PICK_APPROACH_M, pqx, pqy, pqz, pqw)
        if pre_grasp is None:
            self.get_logger().error(f'IK failed: pre-grasp brick {self.brick_idx}')
            return

        grasp = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, px, py, pz + PICK_GRASP_M, pqx, pqy, pqz, pqw)
        if grasp is None:
            self.get_logger().error(f'IK failed: grasp brick {self.brick_idx}')
            return

        retract_pick = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, px, py, pz + PICK_RETRACT_M, pqx, pqy, pqz, pqw)

        pre_place = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, lx, ly, lz + PLACE_APPROACH_M, lqx, lqy, lqz, lqw)
        if pre_place is None:
            self.get_logger().error(f'IK failed: pre-place brick {self.brick_idx}')
            return

        place = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, lx, ly, lz + PLACE_DEPOSIT_M, lqx, lqy, lqz, lqw)
        if place is None:
            self.get_logger().error(f'IK failed: place brick {self.brick_idx}')
            return

        retract_place = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, lx, ly, lz + PLACE_RETRACT_M, lqx, lqy, lqz, lqw)

        self.job_queue.append(pre_grasp)
        self.job_queue.append(grasp)
        self.job_queue.append('toggle_grip')   # close
        if retract_pick:
            self.job_queue.append(retract_pick)
        self.job_queue.append(SAFE_JOINT_STATE)
        self.job_queue.append(pre_place)
        self.job_queue.append(place)
        self.job_queue.append('toggle_grip')   # open
        if retract_place:
            self.job_queue.append(retract_place)

        self.get_logger().info(f'Queue built: {len(self.job_queue)} steps.')

    # ── Job execution ───────────────────────────────────────────────────────────

    def execute_jobs(self):
        if not self.job_queue:
            self._on_queue_done()
            return

        self.get_logger().info(f'{len(self.job_queue)} step(s) remaining.')
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error('Motion planning failed — skipping step.')
                self.execute_jobs()
                return
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif next_job == 'toggle_grip':
            self.get_logger().info('[GRIP] toggle_grip dequeued — spawning gripper thread.')
            self._toggle_gripper()

        else:
            self.get_logger().error(f'Unknown job: {next_job}')
            self.execute_jobs()

    def _toggle_gripper(self):
        import time
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('[GRIP] service not available — aborting.')
            rclpy.shutdown()
            return
        time.sleep(0.5)
        future = self.gripper_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        try:
            result = future.result()
            self.get_logger().info(f'[GRIP] toggled — success={result.success} msg="{result.message}".')
        except Exception as e:
            self.get_logger().error(f'[GRIP] toggle failed: {e}')
        time.sleep(0.5)
        self.execute_jobs()

    def _execute_joint_trajectory(self, joint_traj):
        self.exec_ac.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected.')
            rclpy.shutdown()
            return
        goal_handle.get_result_async().add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')

    # ── Detection callbacks & matching ──────────────────────────────────────────

    def _set_scanning(self, enabled: bool):
        self._scanning_active = enabled
        self._detection_enable_pub.publish(Bool(data=enabled))

    def _detection_pose_cb(self, msg: PoseArray):
        if self._scanning_active:
            self._latest_poses = msg.poses

    def _detection_meta_cb(self, msg: String):
        if not self._scanning_active:
            return
        meta = json.loads(msg.data)
        poses = self._latest_poses
        with self._detection_lock:
            self._latest_detections = [
                {**m, 'pose': poses[i]}
                for i, m in enumerate(meta)
                if i < len(poses)
            ]
        self._detection_event.set()

    def _parse_label(self, label: str):
        """Parse 'blue 2x4' or 'light_green 2x4' → ('blue', (2, 4))."""
        parts = label.lower().split()
        shape = None
        for part in parts:
            if 'x' in part:
                try:
                    r, c = part.split('x', 1)
                    shape = (int(r), int(c))
                except ValueError:
                    pass
                break
        color_parts = [p for p in parts if 'x' not in p]
        color = '_'.join(color_parts)
        if color not in _KNOWN_COLORS:
            color = None
        return color, shape

    def _match_brick(self, label: str) -> dict | None:
        """Return the first detected brick matching the needed color and shape."""
        color, shape = self._parse_label(label)
        if color is None or shape is None:
            self.get_logger().warn(f'Could not parse label "{label}"')
            return None
        needed = tuple(sorted(shape))
        with self._detection_lock:
            for d in self._latest_detections:
                if d['color'] != color:
                    continue
                if tuple(sorted(d['shape'])) == needed:
                    return d
        return None

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        self.joint_state = msg


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bricks', type=Path, required=True,
                        help='Path to bricks JSON file (lego_build.json or bricks_test.json).')
    parser.add_argument('--picks', type=Path, default=None,
                        help='Path to picks JSON file (bricks_test.json). Required for lego_build.json format.')
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = LegoBuilder(bricks_json=parsed.bricks, picks_json=parsed.picks)
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
