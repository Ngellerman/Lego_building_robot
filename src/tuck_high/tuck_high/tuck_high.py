#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint
from control_msgs.action import FollowJointTrajectory


# Overhead observation pose (radians) — higher than the default tuck
JOINTS = {
    'shoulder_pan_joint':  4.716,
    'shoulder_lift_joint': -1.592,
    'elbow_joint':         -0.979,
    'wrist_1_joint':       -2.124,
    'wrist_2_joint':        1.576,
    'wrist_3_joint':       -3.137,
}


class TuckHigh(Node):
    def __init__(self):
        super().__init__('tuck_high')

        self.plan_cli = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        while not self.plan_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /plan_kinematic_path...')

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = 5.0

        goal = Constraints()
        for joint_name, position in JOINTS.items():
            goal.joint_constraints.append(JointConstraint(
                joint_name=joint_name,
                position=position,
                tolerance_above=0.01, tolerance_below=0.01, weight=1.0
            ))
        req.motion_plan_request.goal_constraints.append(goal)

        future = self.plan_cli.call_async(req)
        future.add_done_callback(self._on_plan)

    def _on_plan(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'Planning service failed: {e}')
            rclpy.shutdown()
            return

        mpr = res.motion_plan_response
        err = mpr.error_code.val
        if err != 1 or not mpr.trajectory.joint_trajectory.points:
            self.get_logger().error(f'Planning failed (error_code={err}).')
            rclpy.shutdown()
            return

        self.get_logger().info('Trajectory planned — executing...')
        self._execute_joint_trajectory(mpr.trajectory.joint_trajectory)

    def _execute_joint_trajectory(self, joint_traj):
        self.exec_ac.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            future.result().result
            self.get_logger().info('Tuck high finished.')
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
        finally:
            rclpy.shutdown()


def main():
    rclpy.init()
    node = TuckHigh()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
