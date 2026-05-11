"""
camera_capture_image.py

Utility to capture one RGB frame from the RealSense camera feed.

Usage as a library:
    from camera_capture_image import capture_image
    image = capture_image(node)   # returns PIL.Image

Usage as a script (saves to disk):
    python3 cv/camera_capture_image.py [output.jpg]
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from PIL import Image as PILImage

_bridge = CvBridge()
_TOPIC = '/camera/camera/color/image_raw'


def capture_image(node: Node, timeout_sec: float = 5.0) -> PILImage.Image | None:
    """Return one RGB frame from the camera as a PIL Image, or None on timeout."""
    ok, msg = wait_for_message(RosImage, node, _TOPIC, time_to_wait=timeout_sec)
    if not ok:
        node.get_logger().error(f'No image received on {_TOPIC} within {timeout_sec}s.')
        return None
    cv_img = _bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    return PILImage.fromarray(cv_img)


def main():
    rclpy.init()
    node = Node('camera_capture')
    image = capture_image(node)
    if image is not None:
        out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('capture.jpg')
        image.save(out)
        print(f'Saved {image.width}x{image.height} image to {out}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
