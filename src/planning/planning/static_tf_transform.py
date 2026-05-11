#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class ConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('constant_tf_publisher')
        self.br = StaticTransformBroadcaster(self)

        # G_wrist_3_link -> camera_link
        # Derived from G_wrist_3_link->camera_depth_optical (identity rot, t=[-0.025, 0.13, 0])
        # times inverse of camera_link->camera_depth_optical (rpy=(-pi/2, 0, -pi/2), t=(0,0,0))
        # R_camera_link->depth_optical = [[0,0,1],[-1,0,0],[0,-1,0]]
        # R_wrist->camera_link = R_wrist->depth_optical * R_depth_optical->camera_link = I * R^T
        G = np.array([[ 0, -1,  0, -0.025],
                      [ 0,  0, -1,  0.130],
                      [ 1,  0,  0,  0.000],
                      [ 0,  0,  0,  1.000]])

        q = R.from_matrix(G[:3, :3]).as_quat()  # [x, y, z, w]

        self.transform = TransformStamped()
        self.transform.header.frame_id = 'wrist_3_link'
        self.transform.child_frame_id = 'camera_link'

        self.transform.transform.translation.x = G[0, 3]
        self.transform.transform.translation.y = G[1, 3]
        self.transform.transform.translation.z = G[2, 3]

        self.transform.transform.rotation.x = q[0]
        self.transform.transform.rotation.y = q[1]
        self.transform.transform.rotation.z = q[2]
        self.transform.transform.rotation.w = q[3]

        self.get_logger().info(f"Broadcasting wrist_3_link -> camera_link:\n{G}\nQuaternion: {q}")

        self.timer = self.create_timer(0.05, self.broadcast_tf)

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
