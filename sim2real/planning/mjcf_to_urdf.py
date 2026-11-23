"""Export the kinematic chain of a compiled MuJoCo model as a URDF (for cuRobo, which only reads URDF).

Only the serial chain from `base_body` to the body holding `tool_site` is exported: one link per body,
one joint per link (revolute/prismatic, or fixed for bodies without joints), body inertials, and the tool
site as a fixed child link. No visual or collision geometry, and side branches (e.g. gripper fingers)
are dropped, so the URDF is purely kinematic + inertial. Depends only on numpy and mujoco.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np


@dataclasses.dataclass(frozen=True)
class UrdfInfo:
  path: Path
  base_link: str
  tool_link: str
  joint_names: tuple[str, ...]
  lower: tuple[float, ...]
  upper: tuple[float, ...]
  velocity: tuple[float, ...]


def quat_to_mat(quat_wxyz) -> np.ndarray:
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, np.asarray(quat_wxyz, np.float64))
  return m.reshape(3, 3)


def mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
  """URDF rpy (fixed-axis X then Y then Z, i.e. R = Rz(yaw) Ry(pitch) Rx(roll))."""
  pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
  if abs(np.cos(pitch)) > 1e-9:
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
  else:  # gimbal lock: only yaw - roll (or yaw + roll) is defined
    roll = 0.0
    yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
  return roll, pitch, yaw


def _fmt(v) -> str:
  return " ".join(repr(float(x)) for x in v)


def _origin(pos, quat_wxyz) -> str:
  return f'<origin xyz="{_fmt(pos)}" rpy="{_fmt(mat_to_rpy(quat_to_mat(quat_wxyz)))}"/>'


def _inertial(m: mujoco.MjModel, b: int) -> str:
  mass = float(m.body_mass[b])
  if mass <= 0.0:
    return ""
  ix, iy, iz = (float(x) for x in m.body_inertia[b])
  return (f"<inertial>{_origin(m.body_ipos[b], m.body_iquat[b])}<mass value={quoteattr(repr(mass))}/>"
          f'<inertia ixx="{ix!r}" ixy="0" ixz="0" iyy="{iy!r}" iyz="0" izz="{iz!r}"/></inertial>')


def _chain(m: mujoco.MjModel, base: int, tip: int) -> list[int]:
  bodies = [tip]
  while bodies[-1] != base:
    parent = int(m.body_parentid[bodies[-1]])
    if bodies[-1] == 0:
      raise ValueError(f"body {m.body(base).name!r} is not an ancestor of {m.body(tip).name!r}")
    bodies.append(parent)
  return bodies[::-1]


def export_urdf(
  m: mujoco.MjModel,
  base_body: str,
  tool_site: str,
  joint_names: Sequence[str],
  out_path: str | Path,
  effort: Sequence[float] | float = 100.0,
  velocity: Sequence[float] | float = 2.0,
  robot_name: str = "robot",
) -> UrdfInfo:
  """Write the `base_body` -> `tool_site` chain of `m` to `out_path` as URDF.

  joint_names: the chain's actuated joints in order; must match the joints found on the chain exactly.
  effort / velocity: per-joint (or scalar) URDF limits [Nm or N, rad/s or m/s]; cuRobo uses velocity.
  The tool link is named after `tool_site`, the base link after `base_body`.
  """
  base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body)
  site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tool_site)
  if base < 0:
    raise ValueError(f"no body {base_body!r}")
  if site < 0:
    raise ValueError(f"no site {tool_site!r}")
  joint_names = tuple(joint_names)
  n = len(joint_names)
  effort = np.broadcast_to(np.asarray(effort, np.float64), (n,))
  velocity = np.broadcast_to(np.asarray(velocity, np.float64), (n,))

  chain = _chain(m, base, int(m.site_bodyid[site]))
  links = [f"<link name={quoteattr(m.body(chain[0]).name)}>{_inertial(m, chain[0])}</link>"]
  joints, found, lower, upper = [], [], [], []
  for b in chain[1:]:
    name, parent = m.body(b).name, m.body(int(m.body_parentid[b])).name
    links.append(f"<link name={quoteattr(name)}>{_inertial(m, b)}</link>")
    origin = _origin(m.body_pos[b], m.body_quat[b])
    nj = int(m.body_jntnum[b])
    if nj == 0:
      joints.append(f'<joint name={quoteattr(name + "_fixed")} type="fixed"><parent link={quoteattr(parent)}/>'
                    f"<child link={quoteattr(name)}/>{origin}</joint>")
      continue
    if nj > 1:
      raise ValueError(f"body {name!r} has {nj} joints; URDF needs one joint per link")
    j = int(m.body_jntadr[b])
    jtype = int(m.jnt_type[j])
    if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
      raise ValueError(f"joint {m.joint(j).name!r}: only hinge/slide joints can be exported")
    if np.linalg.norm(m.jnt_pos[j]) > 1e-9:
      raise ValueError(f"joint {m.joint(j).name!r} is offset from its body origin; not supported")
    jname = m.joint(j).name
    if len(found) >= n or jname != joint_names[len(found)]:
      raise ValueError(f"chain joint {jname!r} does not match joint_names {joint_names}")
    k = len(found)
    lo, hi = (float(x) for x in (m.jnt_range[j] if m.jnt_limited[j] else (-2 * np.pi, 2 * np.pi)))
    found.append(jname)
    lower.append(lo)
    upper.append(hi)
    kind = "revolute" if jtype == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
    joints.append(
      f'<joint name={quoteattr(jname)} type="{kind}"><parent link={quoteattr(parent)}/>'
      f'<child link={quoteattr(name)}/>{origin}<axis xyz="{_fmt(m.jnt_axis[j])}"/>'
      f'<limit lower="{lo!r}" upper="{hi!r}" effort="{float(effort[k])!r}" velocity="{float(velocity[k])!r}"/></joint>')
  if tuple(found) != joint_names:
    raise ValueError(f"chain joints {tuple(found)} != joint_names {joint_names}")

  tool = m.site(site).name
  links.append(f"<link name={quoteattr(tool)}/>")
  joints.append(f'<joint name={quoteattr(tool + "_fixed")} type="fixed">'
                f"<parent link={quoteattr(m.body(chain[-1]).name)}/><child link={quoteattr(tool)}/>"
                f"{_origin(m.site_pos[site], m.site_quat[site])}</joint>")

  # The <mujoco> block lets MuJoCo reload this URDF for verification without fusing the fixed tool link.
  xml = "\n".join([
    f"<robot name={quoteattr(robot_name)}>",
    '  <mujoco><compiler fusestatic="false" discardvisual="true"/></mujoco>',
    *(f"  {s}" for s in links + joints),
    "</robot>",
    "",
  ])
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(xml)
  return UrdfInfo(out_path, m.body(base).name, tool, joint_names, tuple(lower), tuple(upper),
                  tuple(float(v) for v in velocity))
