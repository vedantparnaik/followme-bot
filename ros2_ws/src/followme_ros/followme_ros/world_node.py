"""The simulated world: people, obstacles, drivetrain and sensors.

Stands in for the robot. It takes duty commands and produces what the real
robot produces: camera detections, range readings and odometry.

  sub  /cmd_duty          geometry_msgs/Vector3   x = left %, y = right %
  pub  /detections        std_msgs/String (JSON)  what the detector saw
  pub  /range             std_msgs/String (JSON)  range points + free cones
  pub  /odom, TF odom->base_link
  pub  /scan              sensor_msgs/LaserScan   lidar only, for RViz
  pub  /world/markers     visualization_msgs/MarkerArray
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from followme.config import PROFILES
from followme_ros.codec import encode_detections, encode_range
from simkit.plants import PLANTS
from simkit.scenarios import SCENARIOS
from simkit.sensors import SENSORS, Lidar, VirtualCamera
from simkit.sim import footprint
from simkit.world import Circle

PHYS_HZ, SENSE_HZ = 100.0, 30.0
CMD_TIMEOUT_S = 0.5


def quat_z(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def rgb(c):
    """'#rrggbb' or an (r, g, b) tuple in 0..1."""
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(float(v) for v in c[:3])


class WorldNode(Node):
    def __init__(self):
        super().__init__("world")
        self.declare_parameter("scenario", "obstacles")
        self.declare_parameter("plant", "hardware")
        self.declare_parameter("sensor", "sonar3")
        self.declare_parameter("seed", 0)
        self.declare_parameter("loop", True)
        g = lambda n: self.get_parameter(n).value
        self.scenario_name, self.plant_name = g("scenario"), g("plant")
        self.sensor_name, self.seed, self.loop = g("sensor"), int(g("seed")), bool(g("loop"))
        self.cfg = PROFILES["hardware"]()

        self.pub_det = self.create_publisher(String, "/detections", 10)
        self.pub_rng = self.create_publisher(String, "/range", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/world/markers", 10)
        self.pub_reset = self.create_publisher(String, "/follow/cmd", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Vector3, "/cmd_duty", self._on_cmd, 10)

        self._reset()
        self.create_timer(1.0 / PHYS_HZ, self._physics)
        self.create_timer(1.0 / SENSE_HZ, self._sense)
        self.create_timer(0.1, self._markers)
        self.get_logger().info(
            f"scenario={self.scenario_name} plant={self.plant_name} sensor={self.sensor_name}: "
            f"{self.sc.about}")

    def _reset(self):
        self.sc = SCENARIOS[self.scenario_name](self.seed)
        rng = np.random.default_rng(self.seed + 1000)
        self.camera = VirtualCamera(self.cfg.camera, rng, id_swaps=self.sc.id_swaps)
        cls = SENSORS[self.sensor_name]
        self.ranger = cls(rng) if cls else None
        self.plant = PLANTS[self.plant_name](self.cfg.body, *self.sc.start)
        self.t = 0.0
        self.cmd, self.cmd_t = (0.0, 0.0), -1e9

    def _on_cmd(self, msg: Vector3):
        self.cmd, self.cmd_t = (msg.x, msg.y), self.t

    # ------------------------------------------------------------------
    def _physics(self):
        dt = 1.0 / PHYS_HZ
        l, r = self.cmd if self.t - self.cmd_t < CMD_TIMEOUT_S else (0.0, 0.0)
        prev = self.plant.pose
        self.plant.step(l, r, dt, self.t)
        if self._hits():
            self.plant.x, self.plant.y, self.plant.yaw = prev
            self.plant.halt()
        self.t += dt
        if self.t > self.sc.duration + 3.0 and self.loop:
            self._reset()
            self.pub_reset.publish(String(data="relock"))
            self.get_logger().info("scenario restarted")

    def _hits(self) -> bool:
        b = self.cfg.body
        w = self.sc.world
        for px, py in footprint(self.plant.pose, b.length, b.width):
            if any(ob.shape.distance(px, py) < 0 for ob in w.obstacles):
                return True
            if any(c.distance(px, py) < 0 for _, c in w.person_circles()):
                return True
        return False

    def _sense(self):
        w = self.sc.world
        w.t = self.t
        dets = self.camera.observe(w, self.plant.pose, self.t)
        self.pub_det.publish(String(data=encode_detections(self.t, dets)))
        pts = self.ranger.sense(w, self.plant.pose) if self.ranger else []
        cones = self.ranger.cones() if self.ranger else []
        self.pub_rng.publish(String(data=encode_range(self.t, pts, cones)))
        stamp = self.get_clock().now().to_msg()
        self._odom(stamp)
        if isinstance(self.ranger, Lidar):
            self._scan(stamp)

    def _odom(self, stamp):
        x, y, yaw = self.plant.pose
        qx, qy, qz, qw = quat_z(yaw)
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y = x, y
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf.sendTransform(tf)
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, "odom", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = x, y
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = qz, qw
        self.pub_odom.publish(od)

    def _scan(self, stamp):
        rg = self.ranger
        s = LaserScan()
        s.header.stamp, s.header.frame_id = stamp, "base_link"
        s.angle_min, s.angle_max = rg.angles[0], rg.angles[-1]
        s.angle_increment = rg.angles[1] - rg.angles[0]
        s.range_min, s.range_max = 0.02, rg.max_r
        s.ranges = [float(r) for _, r in rg.last]
        self.pub_scan.publish(s)

    # ------------------------------------------------------------------
    def _markers(self):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        mid = 0

        def mk(kind, ns, x, y, z, sx, sy, sz, color, a=1.0, frame="odom"):
            nonlocal mid
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, frame
            m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
            mid += 1
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = float(sx), float(sy), float(sz)
            m.color.r, m.color.g, m.color.b = (float(v) for v in color)
            m.color.a = float(a)
            arr.markers.append(m)
            return m

        for ob in self.sc.world.obstacles:
            s, h, col = ob.shape, ob.height, rgb(ob.color)
            if isinstance(s, Circle):
                mk(Marker.CYLINDER, "obstacles", s.x, s.y, h / 2, 2 * s.r, 2 * s.r, h, col)
            else:
                mk(Marker.CUBE, "obstacles", s.x, s.y, h / 2, s.sx, s.sy, h, col)
            t = mk(Marker.TEXT_VIEW_FACING, "labels", s.x, s.y, h + 0.25, 0, 0, 0.22,
                   (0.9, 0.9, 0.9))
            t.text = ob.name + ("" if ob.camera_label else " (camera-blind)")
        for p, c in self.sc.world.person_circles():
            col = rgb(p.color)
            mk(Marker.CYLINDER, "people", c.x, c.y, p.height / 2, 2 * c.r, 2 * c.r, p.height, col)
            t = mk(Marker.TEXT_VIEW_FACING, "labels", c.x, c.y, p.height + 0.25, 0, 0, 0.25, col)
            t.text = p.name
        b = self.cfg.body
        x, y, yaw = self.plant.pose
        body = mk(Marker.CUBE, "rover", x, y, 0.15, b.length, b.width, 0.3, (0.95, 0.75, 0.2))
        _, _, body.pose.orientation.z, body.pose.orientation.w = quat_z(yaw)
        self.pub_mk.publish(arr)


def main():
    rclpy.init()
    node = WorldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
