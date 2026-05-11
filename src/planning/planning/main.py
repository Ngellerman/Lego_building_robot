# ROS Libraries
from std_srvs.srv import Trigger
import argparse
import json
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
import subprocess
import threading
from planning.ik import IKPlanner

_LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

ARTIFACTS_DIR = Path(
    '/home/cc/ee106a/sp26/class/ee106a-acz/Desktop/eecs106a-FINAL (segmentation pipeline)/artifacts_cv'
)

DROP_OFF_POSITION = [
    4.492536544799805, -1.4712893974832078, -1.4076955318450928,
    -1.6918603382506312, 0.5925133228302002, -3.1838446299182337,
]

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


def _canonical_footwear_type(label: str) -> str:
    label = label.lower()
    for t in ['sneaker', 'flip flop', 'slipper', 'shoe']:
        if t in label:
            return t
    return label


def _load_shoe_labels() -> list[str]:
    """Return ordered list of canonical footwear types indexed by pub_idx."""
    approved = json.loads((ARTIFACTS_DIR / 'approved_segments.json').read_text())
    approved_ids = set(approved.get('approved_ids', []))
    overrides = {int(k): v for k, v in approved.get('label_overrides', {}).items()}
    segments = json.loads((ARTIFACTS_DIR / 'segments.json').read_text())['segments']

    labels = []
    for seg in segments:
        if seg['id'] not in approved_ids:
            continue
        raw = overrides.get(seg['id'], seg['label'])
        labels.append(_canonical_footwear_type(raw))
    return labels


def _make_joint_state(names: list[str], positions: list[float]) -> JointState:
    js = JointState()
    js.name = names
    js.position = positions
    return js


class UR7e_MultiGrasp(Node):
    def __init__(self, pick_order: list[int]):
        super().__init__('multi_grasp')
        self.pick_order = pick_order
        self.shoe_labels = _load_shoe_labels()

        self.get_logger().info(f'Shoe labels by pub_idx: {list(enumerate(self.shoe_labels))}')
        self.get_logger().info(f'Pick order: {pick_order}')

        self.joint_state = None
        self.grab_points = {}   # pub_idx -> (x, y, z, qx, qy, qz, qw)
        self.job_queue = []
        self._queue_built = False

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 1)
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.get_logger().info('Loading IK planner...')
        self.ik_planner = IKPlanner()
        self.get_logger().info('IK planner ready.')

        for idx in pick_order:
            if idx >= len(self.shoe_labels):
                self.get_logger().error(
                    f'Shoe idx {idx} out of range (only {len(self.shoe_labels)} approved shoes).')
                continue
            if self.shoe_labels[idx] == 'sneaker':
                self.create_subscription(
                    PointStamped, f'/shoe_heel_point_{idx}',
                    lambda msg, i=idx: self._on_heel_point(i, msg),
                    _LATCHED_QOS)
                self.get_logger().info(f'  Subscribed to /shoe_heel_point_{idx} (sneaker)')
            else:
                self.create_subscription(
                    PoseStamped, f'/shoe_goal_pose_{idx}',
                    lambda msg, i=idx: self._on_goal_pose(i, msg),
                    _LATCHED_QOS)
                self.get_logger().info(
                    f'  Subscribed to /shoe_goal_pose_{idx} ({self.shoe_labels[idx]})')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        self.joint_state = msg
        self._check_ready()

    def _on_heel_point(self, idx: int, msg: PointStamped):
        if idx in self.grab_points:
            return
        self.grab_points[idx] = (
            msg.point.x, msg.point.y, msg.point.z,
            0.0, 1.0, 0.0, 0.0,  # straight-down for sneaker heel grab
        )
        self.get_logger().info(
            f'Heel point {idx}: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})')
        self._check_ready()

    def _on_goal_pose(self, idx: int, msg: PoseStamped):
        if idx in self.grab_points:
            return
        p = msg.pose
        self.grab_points[idx] = (
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
        )
        self.get_logger().info(
            f'Goal pose {idx}: ({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f})')
        self._check_ready()

    def _check_ready(self):
        if self._queue_built:
            return
        if self.joint_state is None:
            return
        if not set(self.pick_order).issubset(self.grab_points.keys()):
            return
        self._queue_built = True
        self._build_queue()
        if self.job_queue:
            threading.Thread(target=self.execute_jobs, daemon=True).start()

    # ── Job queue construction ─────────────────────────────────────────────────

    def _build_queue(self):
        self.get_logger().info('Building job queue...')

        drop_js = _make_joint_state(JOINT_NAMES, DROP_OFF_POSITION)

        for shoe_num, idx in enumerate(self.pick_order):
            x, y, z, qx, qy, qz, qw = self.grab_points[idx]
            label = self.shoe_labels[idx]
            self.get_logger().info(
                f'  [{shoe_num + 1}/{len(self.pick_order)}] idx={idx} ({label}) '
                f'@ ({x:.3f}, {y:.3f}, {z:.3f})')

            # 1) Tuck via ros2 run tuck_high tuck_high
            self.job_queue.append('tuck')

            # 2) Pre-grasp (above shoe)
            pre_grasp = self.ik_planner.compute_ik(
                self.joint_state, x, y, z + 0.185, qx, qy, qz, qw)
            if pre_grasp is None:
                self.get_logger().error(f'  IK failed for shoe {idx} pre-grasp — skipping shoe.')
                self.job_queue.pop()  # remove tuck too
                continue
            self.job_queue.append(pre_grasp)

            # 3) Grasp (lower to shoe)
            grasp = self.ik_planner.compute_ik(
                self.joint_state, x, y, z + 0.14, qx, qy, qz, qw)
            if grasp is None:
                self.get_logger().error(f'  IK failed for shoe {idx} grasp — skipping shoe.')
                self.job_queue.pop(); self.job_queue.pop()
                continue
            self.job_queue.append(grasp)

            # 4) Close gripper
            self.job_queue.append('toggle_grip')

            # 5) Retract high
            retract = self.ik_planner.compute_ik(
                self.joint_state, x, y, z + 0.8, qx, qy, qz, qw)
            if retract is None:
                self.get_logger().warn(f'  IK failed for shoe {idx} retract — skipping retract.')
            else:
                self.job_queue.append(retract)

            # 6) Move to hardcoded drop-off
            self.job_queue.append(drop_js)

            # 7) Release gripper
            self.job_queue.append('toggle_grip')

        self.get_logger().info(f'Job queue ready: {len(self.job_queue)} steps.')

    # ── Job execution (callback-driven) ────────────────────────────────────────

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info('All jobs complete. Shutting down.')
            rclpy.shutdown()
            return

        self.get_logger().info(f'{len(self.job_queue)} job(s) remaining.')
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            self.get_logger().info(
                f'Planning to: {list(zip(next_job.name, [round(p, 3) for p in next_job.position]))}')
            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error('Motion planning failed — skipping step.')
                self.execute_jobs()
                return
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif next_job == 'tuck':
            self.get_logger().info('Tucking (ros2 run tuck_high tuck_high)...')
            def _do_tuck():
                import time; time.sleep(1.0)  # let controller settle after previous trajectory
                subprocess.run(['ros2', 'run', 'ur7e_utils', 'tuck'], check=True)
                self.execute_jobs()
            threading.Thread(target=_do_tuck, daemon=True).start()

        elif next_job == 'toggle_grip':
            self.get_logger().info('Toggling gripper...')
            self._toggle_gripper()

        else:
            self.get_logger().error(f'Unknown job type: {next_job}')
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


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--order', type=int, nargs='+', required=True,
        help='Ordered pub_idx list of shoes to pick. E.g. --order 0 1 2')
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = UR7e_MultiGrasp(pick_order=parsed.order)
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
