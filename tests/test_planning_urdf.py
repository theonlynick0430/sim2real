"""Planning-module pieces that run anywhere (no torch / cuRobo / GPU): URDF export, host FK, frames, config."""

import ast
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim2real.planning import MujocoFK, PlannerConfig, base_to_world, rotation_error, world_to_base
from sim2real.planning.adapters import export_robot_urdf, planner_config_from_robot
from sim2real.robots import get_robot
from sim2real.sim.scene import robot_spec

PLANNING = Path(__file__).resolve().parents[1] / "sim2real" / "planning"


@pytest.fixture(scope="module")
def yam_urdf(tmp_path_factory):
  return export_robot_urdf(get_robot("yam"), tmp_path_factory.mktemp("urdf"))


def _mjcf_tcp(robot, q):
  """TCP pose in the robot root-body frame, from the original MJCF."""
  m = robot_spec(robot).compile()
  d = mujoco.MjData(m)
  for k, j in enumerate(robot.arm_joints):
    d.qpos[m.joint(j).qposadr[0]] = q[k]
  mujoco.mj_kinematics(m, d)
  base, site = m.body("arm").id, m.site(robot.tcp_site).id
  Rb = d.xmat[base].reshape(3, 3)
  pos = Rb.T @ (d.site_xpos[site] - d.xpos[base])
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, (Rb.T @ d.site_xmat[site].reshape(3, 3)).ravel())
  return pos, quat


def test_urdf_fk_matches_mjcf(yam_urdf):
  robot = get_robot("yam")
  fk = MujocoFK(yam_urdf.path, yam_urdf.joint_names, yam_urdf.tool_link)
  qs = fk.sample_q(200, np.random.default_rng(0), margin=0.0)
  pos, quat = fk(qs)
  for i, q in enumerate(qs):
    p_ref, q_ref = _mjcf_tcp(robot, q)
    np.testing.assert_allclose(pos[i], p_ref, atol=1e-9)
    assert rotation_error(quat[i], q_ref) < 1e-6


def test_urdf_metadata(yam_urdf):
  robot = get_robot("yam")
  m = robot_spec(robot).compile()
  assert yam_urdf.base_link == "arm" and yam_urdf.tool_link == robot.tcp_site
  assert yam_urdf.joint_names == robot.arm_joints
  ranges = np.array([m.joint(j).range for j in robot.arm_joints])
  np.testing.assert_allclose(yam_urdf.lower, ranges[:, 0])
  np.testing.assert_allclose(yam_urdf.upper, ranges[:, 1])
  u = mujoco.MjModel.from_xml_path(str(yam_urdf.path))
  assert u.nq == robot.ndof  # gripper branch dropped
  chain = ["arm", *(f"link_{i}" for i in range(1, 7))]
  np.testing.assert_allclose(sum(u.body(b).mass[0] for b in chain), sum(m.body(b).mass[0] for b in chain), rtol=1e-9)


def test_planner_config(tmp_path):
  cfg = planner_config_from_robot(get_robot("yam"), tmp_path, velocity=1.5, max_batch_size=8)
  assert cfg.dof == 6 and cfg.max_vel == (1.5,) * 6 and cfg.max_batch_size == 8
  kin = cfg.curobo_robot_dict()["kinematics"]
  assert "collision_spheres" not in kin and kin["tool_frames"] == ["pinch_site"]
  assert kin["cspace"]["joint_names"] == list(cfg.joint_names)
  assert Path(kin["urdf_path"]).is_file()
  with pytest.raises(ValueError):
    PlannerConfig("x.urdf", "b", "t", ("j1", "j2"), default_q=(0.0,))


def test_frames_roundtrip():
  rng = np.random.default_rng(1)
  base_pos, base_quat = rng.normal(size=3), rng.normal(size=4)
  base_quat /= np.linalg.norm(base_quat)
  pos, quat = rng.normal(size=(10, 3)), rng.normal(size=(10, 4))
  quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
  pb, qb = world_to_base(pos, quat, base_pos, base_quat)
  pw, qw = base_to_world(pb, qb, base_pos, base_quat)
  np.testing.assert_allclose(pw, pos, atol=1e-12)
  assert rotation_error(qw, quat).max() < 1e-6
  # A pose at the base itself maps to the identity.
  p0, q0 = world_to_base(base_pos, base_quat, base_pos, base_quat)
  np.testing.assert_allclose(p0, 0, atol=1e-12)
  assert rotation_error(q0, [1, 0, 0, 0]) < 1e-6


def test_planning_is_self_contained():
  """Only adapters.py may import the rest of sim2real; everything else uses relative imports."""
  for f in PLANNING.glob("*.py"):
    if f.name == "adapters.py":
      continue
    for node in ast.walk(ast.parse(f.read_text())):
      if isinstance(node, ast.ImportFrom) and node.level == 0:
        assert not (node.module or "").startswith("sim2real"), f"{f.name} imports {node.module}"
      if isinstance(node, ast.Import):
        assert not any(a.name.startswith("sim2real") for a in node.names), f"{f.name} imports sim2real"
