"""The follow-me brain as a ROS 2 node. Same ``followme.Brain`` as the robot.

  sub  /detections   std_msgs/String (JSON)   one brain step per message
  sub  /range        std_msgs/String (JSON)   latest range points + free cones
  sub  /odom         nav_msgs/Odometry        for obstacle memory and PURSUE
  sub  /follow/cmd   std_msgs/String          arm | disarm | estop | reset | relock
  pub  /cmd_duty     geometry_msgs/Vector3    x = left %, y = right %
  pub  /follow/state std_msgs/String (JSON)   state, note, event, target, fix
  pub  /follow/markers visualization_msgs/MarkerArray (base_link)
"""
from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from followme.brain import Brain, State
from followme.config import PROFILES
from followme_ros.codec import decode_detections, decode_range

RANGE_STALE_S = 0.3

STATE_RGB = {
    State.IDLE: (0.6, 0.6, 0.6), State.ACQUIRE: (0.9, 0.9, 0.3), State.FOLLOW: (0.3, 0.9, 0.4),
    State.AVOID: (1.0, 0.65, 0.2), State.BLOCKED: (1.0, 0.3, 0.2), State.PURSUE: (0.2, 0.8, 0.9),
    State.SEARCH: (0.2, 0.6, 1.0), State.WAIT: (0.6, 0.4, 0.9), State.ESTOP: (1.0, 0.0, 0.2),
}


class BrainNode(Node):
    def __init__(self):
        super().__init__("brain")
        self.declare_parameter("profile", "hardware")
        self.declare_parameter("auto_arm", True)
        self.brain = Brain(PROFILES[self.get_parameter("profile").value]())
        if self.get_parameter("auto_arm").value:
            self.brain.arm()
        self.range = (-1e9, [], [])
        self.odom = None
        self.pub_cmd = self.create_publisher(Vector3, "/cmd_duty", 10)
        self.pub_state = self.create_publisher(String, "/follow/state", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/follow/markers", 10)
        self.create_subscription(String, "/detections", self._on_dets, 10)
        self.create_subscription(String, "/range", self._on_range, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(String, "/follow/cmd", self._on_cmd, 10)
        self._last_state = None

    def _on_range(self, msg: String):
        self.range = decode_range(msg.data)

    def _on_odom(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odom = (p.x, p.y, 2.0 * math.atan2(q.z, q.w))

    def _on_cmd(self, msg: String):
        cmd = msg.data.strip().lower()
        action = {"arm": self.brain.arm, "disarm": self.brain.disarm, "estop": self.brain.estop,
                  "reset": self.brain.reset, "relock": self.brain.relock}.get(cmd)
        if action is None:
            self.get_logger().warn(f"unknown command {cmd!r}")
            return
        action()
        self.get_logger().info(f"command: {cmd}")

    def _on_dets(self, msg: String):
        t, dets = decode_detections(msg.data)
        rt, pts, cones = self.range
        if t - rt > RANGE_STALE_S:
            pts, cones = [], []
        out = self.brain.step(t, dets, pts, odom=self.odom, free=cones)
        self.pub_cmd.publish(Vector3(x=float(out.l), y=float(out.r), z=0.0))
        fix = out.fix
        self.pub_state.publish(String(data=json.dumps({
            "t": round(t, 2), "state": out.state.value, "note": out.note, "event": out.event,
            "target_id": out.target_id, "how": out.how, "reid": round(out.reid_score, 3),
            "range_m": None if fix is None else round(fix.range_m, 2),
            "bearing_deg": None if fix is None else round(math.degrees(fix.bearing), 1),
            "l": round(out.l, 1), "r": round(out.r, 1)})))
        if out.event:
            self.get_logger().info(f"{out.state.value}: {out.event}")
        elif out.state != self._last_state:
            self.get_logger().info(f"-> {out.state.value}  {out.note}")
        self._last_state = out.state
        self._markers(out)

    def _markers(self, out):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        def mk(kind, ns, mid, rgb, sx=0.05, sy=0.05, sz=0.05, a=1.0):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, "base_link"
            m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = float(sx), float(sy), float(sz)
            m.color.r, m.color.g, m.color.b = (float(v) for v in rgb)
            m.color.a = float(a)
            arr.markers.append(m)
            return m

        pts = mk(Marker.POINTS, "obstacles", 0, (1.0, 0.35, 0.35), 0.06, 0.06)
        pts.points = [Point(x=p.x, y=p.y, z=0.05) for p in out.obstacles]

        fan = mk(Marker.LINE_LIST, "fan", 0, (0.5, 0.5, 0.5), 0.01, a=0.5)
        best = mk(Marker.ARROW, "heading", 0, STATE_RGB[out.state], 0.05, 0.1, 0.1)
        if out.plan is not None:
            for h, clr in out.plan.fan:
                fan.points += [Point(x=0.31 * math.cos(h), y=0.31 * math.sin(h), z=0.02),
                               Point(x=(0.31 + clr) * math.cos(h), y=(0.31 + clr) * math.sin(h),
                                     z=0.02)]
            h, L = out.plan.heading, 0.31 + max(0.3, out.plan.clearance)
            best.points = [Point(x=0.0, y=0.0, z=0.35),
                           Point(x=L * math.cos(h), y=L * math.sin(h), z=0.35)]
        else:
            best.action = Marker.DELETE

        tgt = mk(Marker.SPHERE, "target", 0, (0.2, 1.0, 0.4), 0.3, 0.3, 0.3, a=0.8)
        if out.fix is not None:
            tgt.pose.position.x = out.fix.range_m * math.cos(out.fix.bearing)
            tgt.pose.position.y = out.fix.range_m * math.sin(out.fix.bearing)
            tgt.pose.position.z = 1.9
        else:
            tgt.action = Marker.DELETE

        txt = mk(Marker.TEXT_VIEW_FACING, "state", 0, STATE_RGB[out.state], sz=0.3)
        txt.pose.position.z = 0.9
        txt.text = out.state.value + ("" if out.target_id is None else f" #{out.target_id}")
        self.pub_mk.publish(arr)


def main():
    rclpy.init()
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
