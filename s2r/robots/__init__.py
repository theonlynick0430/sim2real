from s2r.robots.config import ActuatorMode, RobotConfig, get_robot, register_robot, registered_robots
from s2r.robots import yam  # noqa: F401  (registers built-in robots)

__all__ = ["ActuatorMode", "RobotConfig", "get_robot", "register_robot", "registered_robots"]
