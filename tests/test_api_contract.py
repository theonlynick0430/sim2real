"""Pins the MuJoCo / MJWarp API surface sim2real depends on. If this fails after a version bump,
fix the adapters (sim/*_backend.py, dynamics/kernels.py) before anything else."""

import dataclasses
import inspect

import mujoco
import mujoco_warp as mjw
import numpy as np


def test_versions_match():
  assert mujoco.__version__ == mjw.__version__, (mujoco.__version__, mjw.__version__)


def test_mujoco_c_api():
  mjm = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><joint/><geom size='.1'/><site name='s'/></body></worldbody></mujoco>")
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  assert hasattr(mjd, "M") and not hasattr(mjd, "qM")  # 3.17: qM -> M (CSR, nC)
  assert mjd.M.shape == (mjm.nC,)
  for f in ("M_rowadr", "M_colind", "M_rownnz"):
    assert hasattr(mjm, f)
  full = np.zeros((mjm.nv, mjm.nv))
  mujoco.mj_fullM(mjm, mjd, full)  # signature (m, d, dst)
  jacp, jacr = np.zeros((3, mjm.nv)), np.zeros((3, mjm.nv))
  mujoco.mj_jacSite(mjm, mjd, jacp, jacr, 0)
  for fn in ("mj_step1", "mj_step2", "mj_resetDataKeyframe"):
    assert hasattr(mujoco, fn)


def test_mjwarp_api():
  for fn in ("put_model", "make_data", "step1", "step2", "forward", "kinematics", "jac", "reset_data_keyframe",
             "get_data_into"):
    assert hasattr(mjw, fn), fn
  data_fields = {f.name for f in dataclasses.fields(mjw.Data)}
  for f in ("qpos", "qvel", "ctrl", "M", "qfrc_bias", "site_xpos", "site_xmat", "xpos", "mocap_pos", "nacon", "nefc"):
    assert f in data_fields, f
  assert "qM" not in data_fields
  assert list(inspect.signature(mjw.jac).parameters) == ["m", "d", "jacp", "jacr", "point", "body"]
  assert "nconmax" in inspect.signature(mjw.make_data).parameters
