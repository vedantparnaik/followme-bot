"""followme - the brain of a person-following robot.

Pure Python + numpy. No ROS, no OpenCV, no detector. Feed it detections and
obstacle points, get left/right motor duty back.
"""
from followme.brain import Brain, Output, State
from followme.config import Config, hardware, ideal
from followme.obstacles import ObstaclePoint
from followme.perception import Detection

__all__ = ["Brain", "Output", "State", "Config", "hardware", "ideal",
           "ObstaclePoint", "Detection"]
__version__ = "0.1.0"
