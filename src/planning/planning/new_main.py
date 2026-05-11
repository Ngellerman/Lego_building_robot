# ROS Libraries
from std_srvs.srv import Trigger
import argparse
import json
from enum import Enum, auto
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import tf2_ros
import tf2_geometry_msgs
import threading
from planning.ik import IKPlanner


# TODO: replace with real Cartesian poses (x, y, z, qx, qy, qz, qw) in base_link frame
# Default orientation (0, 1, 0, 0) points the gripper straight down.
ARUCO_SCAN_POSE = (-0.4, 0.4, 0.288, 0.0, 1.0, 0.0, 0.0)
BRICK_SCAN_POSE = (0.4, 0.4, 0.288, 0.0, 1.0, 0.0, 0.0)

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



def _load_bricks(json_path: Path) -> list[dict]:
    """
    Expected JSON format:
    {
      "bricks": [
        {
          "id": 0,
          "label": "2x4 red",
          "pick":  {"x": 0.3, "y": 0.2,  "z": 0.05, "qx": 0.0, "qy": 1.0, "qz": 0.0, "qw": 0.0},
          "place": {"x": 0.4, "y": -0.3, "z": 0.05, "qx": 0.0, "qy": 1.0, "qz": 0.0, "qw": 0.0}
        }
      ]
    }
    """
    return json.loads(json_path.read_text())['bricks']


class LegoBuilder(Node):
    def __init__(self, bricks_json: Path):
        super().__init__('lego_builder')

        self.bricks    = _load_bricks(bricks_json)
        self.brick_idx = 0
        self.phase     = Phase.MOVE_TO_ARUCO

        self.joint_state  = None
        self.job_queue    = []
        self.baseplate_tf = None

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f'Loaded {len(self.bricks)} brick(s) from {bricks_json}')

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 1)
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
            # SIMULATED: brick poses already loaded from JSON
            remaining = len(self.bricks) - self.brick_idx
            self.get_logger().info(f'[SIMULATE] Brick scan confirmed. {remaining} brick(s) remaining.')
            self.phase = Phase.PICK_AND_PLACE
            self._advance()

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
        # TODO: re-enable transform once baseplate_frame axes are confirmed
        # if pose_dict.get('frame') == 'baseplate':
        #     return self._transform_to_base(x, y, z, qx, qy, qz, qw)
        return x, y, z, qx, qy, qz, qw

    def _build_pick_place_queue(self):
        brick = self.bricks[self.brick_idx]
        pk    = brick['pick']
        pl    = brick['place']
        label = brick.get('label', f'brick_{self.brick_idx}')

        px, py, pz, pqx, pqy, pqz, pqw = self._resolve_pose(pk)
        lx, ly, lz, lqx, lqy, lqz, lqw = self._resolve_pose(pl)

        self.get_logger().info(
            f'[Brick {self.brick_idx}] {label} | '
            f'pick=({px:.3f},{py:.3f},{pz:.3f}) place=({lx:.3f},{ly:.3f},{lz:.3f})')

        pre_grasp = self.ik_planner.compute_ik(
            self.joint_state, px, py, pz + 0.08, pqx, pqy, pqz, pqw)
        if pre_grasp is None:
            self.get_logger().error(f'IK failed: pre-grasp brick {self.brick_idx}')
            return

        grasp = self.ik_planner.compute_ik(
            self.joint_state, px, py, pz + 0.025, pqx, pqy, pqz, pqw)
        if grasp is None:
            self.get_logger().error(f'IK failed: grasp brick {self.brick_idx}')
            return

        retract_pick = self.ik_planner.compute_ik(
            self.joint_state, px, py, pz + 0.15, pqx, pqy, pqz, pqw)

        pre_place = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, lx, ly, lz + 0.08, lqx, lqy, lqz, lqw)
        if pre_place is None:
            self.get_logger().error(f'IK failed: pre-place brick {self.brick_idx}')
            return

        place = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, lx, ly, lz + 0.033, lqx, lqy, lqz, lqw)
        if place is None:
            self.get_logger().error(f'IK failed: place brick {self.brick_idx}')
            return

        retract_place = self.ik_planner.compute_ik(
            SAFE_JOINT_STATE, lx, ly, lz + 0.15, lqx, lqy, lqz, lqw)

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
            self._toggle_gripper()

        else:
            self.get_logger().error(f'Unknown job: {next_job}')
            self.execute_jobs()

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available.')
            rclpy.shutdown()
            return
        future = self.gripper_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.get_logger().info('Gripper toggled.')
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

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        self.joint_state = msg


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bricks', type=Path, required=True,
                        help='Path to bricks JSON file.')
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = LegoBuilder(bricks_json=parsed.bricks)
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
