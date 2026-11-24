"""On-disk layout of a profile dataset: one npz per trajectory plus `session.json`, with resume.

  <out>/traj_00042.npz   t (T,), q / dq / tau_motor / tau_ext (T, n), skill (T,), meta (JSON string)
  <out>/session.json     generation config, status, one entry per trajectory, the error if a run failed

Files are written to a temporary name and renamed, so a killed run never leaves a half-written file.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import traceback
from pathlib import Path

import numpy as np

SESSION_FILE = "session.json"
ARRAYS = ("t", "q", "dq", "tau_motor", "tau_ext", "skill")


def traj_path(out: Path, index: int) -> Path:
  return Path(out) / f"traj_{index:05d}.npz"


def save_trajectory(path: Path, arrays: dict, meta: dict) -> None:
  tmp = path.with_name(path.name + ".tmp")
  with open(tmp, "wb") as f:
    np.savez_compressed(f, **{k: arrays[k] for k in ARRAYS}, meta=json.dumps(meta))
  os.replace(tmp, path)


def load_trajectory(path: Path) -> tuple[dict, dict]:
  """(arrays, meta) of one trajectory file."""
  with np.load(path) as data:
    return {k: data[k] for k in ARRAYS}, json.loads(str(data["meta"]))


def _now() -> str:
  return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_commit() -> str | None:
  try:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                          check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


class Session:
  """Bookkeeping for one output directory. `config` is everything that determines the generated data; a
  resumed run must use the same one (the number of trajectories may grow)."""

  def __init__(self, out: Path, config: dict, resume: bool = False):
    self.out, self.path = Path(out), Path(out) / SESSION_FILE
    config = json.loads(json.dumps(config))  # as it will read back (tuples -> lists)
    if resume:
      if not self.path.exists():
        raise FileNotFoundError(f"--resume: no {SESSION_FILE} in {self.out}")
      self.data = json.loads(self.path.read_text())
      if self.data["config"] != config:
        changed = sorted(k for k in set(config) | set(self.data["config"])
                         if config.get(k) != self.data["config"].get(k))
        raise ValueError(f"--resume: generation config differs from the saved session in {changed}")
    else:
      self.out.mkdir(parents=True, exist_ok=True)
      if any(self.out.glob("traj_*.npz")) or self.path.exists():
        raise FileExistsError(f"{self.out} already holds a dataset; pass --resume or choose another --out")
      self.data = {"version": 1, "config": config, "created": _now(), "runs": [], "trajectories": {}}
    self.run = {"started": _now(), "ended": None, "git_commit": _git_commit(), "status": "running", "error": None}
    self.data["runs"].append(self.run)
    self.write()

  def done(self, index: int) -> bool:
    """Trajectory `index` is on disk and loads. A listed but missing / corrupt file counts as not done."""
    path = traj_path(self.out, index)
    if not path.exists():
      return False
    try:
      load_trajectory(path)
      return True
    except Exception:  # truncated or corrupt: regenerate
      return False

  def add(self, index: int, arrays: dict, meta: dict, rejects: list[str]) -> None:
    path = traj_path(self.out, index)
    save_trajectory(path, arrays, meta)
    self.data["trajectories"][str(index)] = {
      "file": path.name, "status": "ok", "attempt": meta["attempt"], "num_samples": meta["num_samples"],
      "duration_s": round(meta["num_samples"] * meta["dt"], 4), "skills": [s["type"] for s in meta["skills"]],
      "payload_masses": [round(s["payload_mass"], 4) for s in meta["skills"]],
      "rejects": rejects,
    }
    self.write()

  def fail(self, index: int, rejects: list[str]) -> None:
    """Every attempt at `index` was rejected; it stays missing and a later --resume tries again."""
    self.data["trajectories"][str(index)] = {"file": None, "status": "rejected", "rejects": rejects}
    self.write()

  def close(self, status: str, error: BaseException | None = None) -> None:
    self.run.update(status=status, ended=_now())
    if error is not None:
      self.run["error"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    self.write()

  def write(self) -> None:
    entries = self.data["trajectories"].values()
    self.data["status"] = self.run["status"]
    self.data["updated"] = _now()
    self.data["num_ok"] = sum(e["status"] == "ok" for e in entries)
    self.data["num_rejected"] = sum(e["status"] == "rejected" for e in entries)
    self.data["trajectories"] = dict(sorted(self.data["trajectories"].items(), key=lambda kv: int(kv[0])))
    tmp = self.path.with_name(self.path.name + ".tmp")
    tmp.write_text(json.dumps(self.data, indent=2))
    os.replace(tmp, self.path)
