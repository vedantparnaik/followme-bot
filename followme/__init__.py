"""Brain of a person-following robot.

Takes detections and obstacle points, returns left/right motor duty. Only
needs numpy.
"""
from followme.brain import Brain, Output, State
from followme.config import Config, hardware, ideal
from followme.obstacles import ObstaclePoint
from followme.perception import Detection

__all__ = ["Brain", "Output", "State", "Config", "hardware", "ideal",
           "ObstaclePoint", "Detection"]
__version__ = "0.1.0"
