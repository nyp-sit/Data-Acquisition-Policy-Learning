#!/usr/bin/env python
"""
Run a policy trained in MakerMods Lab on an SO-101 follower arm with plain LeRobot Python.

MakerMods Lab trains with LeRobot's own trainer, so every checkpoint it saves is an ordinary
LeRobot `pretrained_model/` folder. This script loads that folder and drives the arm in a simple
observe -> predict -> act loop, so learners can see (and edit) every step of inference.

Where to point --policy (any one of these works):
  1. A local MakerMods Lab run folder (uses its latest checkpoint):
       ~/.cache/huggingface/lerobot/outputs/train/<policy>_<dataset>_<timestamp>
  2. A specific checkpoint folder:
       .../checkpoints/020000/pretrained_model
  3. The zip from the Lab's "Download this checkpoint" button:   act_pick_step_20000.zip
  4. A Hub repo published from the Lab ("Publish to Hub") or trained on HF cloud:
       your-hf-user/act-pick                 (latest checkpoint in the repo)
       your-hf-user/act-pick@20000           (a specific step)

Examples:
  python run_makermods_policy.py --list
  python run_makermods_policy.py --policy your-hf-user/act-pick --dry-run
  python run_makermods_policy.py --policy your-hf-user/act-pick \
      --port /dev/ttyACM0 --robot-id my_follower \
      --camera front=0 --camera wrist=1 --task "Pick up the cube" --duration 30

Tested against LeRobot 0.6.x (MakerMods Lab pins makermods-robotics/lerobot @ 5ce2fe41a, v0.6.2 base).
SO-101 single arm only. Not covered here: bimanual SO-101, Maker/Metal arms, RTC inference, remote GPU.
"""

from __future__ import annotations

import argparse
import logging
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

# Where MakerMods Lab keeps things on the machine that ran it (LeRobot's default cache).
LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"
LAB_RUNS_DIR = LEROBOT_HOME / "outputs" / "train"
FOLLOWER_CALIB_DIR = LEROBOT_HOME / "calibration" / "robots" / "so_follower"

SO101_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

log = logging.getLogger("run_policy")


# --------------------------------------------------------------------------------------
# Step 1 - find the checkpoint
# --------------------------------------------------------------------------------------
def list_local_runs() -> None:
    """Print every MakerMods Lab run and calibration file on this machine."""
    print(f"Training runs in {LAB_RUNS_DIR}:")
    runs = sorted(p for p in LAB_RUNS_DIR.glob("*") if (p / "checkpoints").is_dir()) if LAB_RUNS_DIR.is_dir() else []
    if not runs:
        print("  (none found)")
    for run in runs:
        steps = sorted(d.name for d in (run / "checkpoints").iterdir() if d.name.isdigit())
        print(f"  {run.name}\n      checkpoints: {', '.join(steps) or '(none yet)'}")
    print(f"\nFollower calibrations in {FOLLOWER_CALIB_DIR} (use the name as --robot-id):")
    calibs = sorted(FOLLOWER_CALIB_DIR.glob("*.json")) if FOLLOWER_CALIB_DIR.is_dir() else []
    print("\n".join(f"  {c.stem}" for c in calibs) or "  (none found)")


def _pick_step_dir(checkpoints: Path, step: str | None) -> Path:
    """Choose checkpoints/<step>/ - the requested step, else the newest one."""
    steps = {int(d.name): d for d in checkpoints.iterdir() if d.name.isdigit()}
    if not steps:
        raise FileNotFoundError(f"No numbered checkpoints inside {checkpoints}")
    if step is None:
        return steps[max(steps)]
    if int(step) not in steps:
        raise FileNotFoundError(f"Step {step} not found. Available: {sorted(steps)}")
    return steps[int(step)]


def resolve_policy_path(spec: str) -> Path:
    """Turn whatever the user passed into a local folder that holds config.json + model.safetensors."""
    path = Path(spec).expanduser()

    if path.is_file() and path.suffix == ".zip":  # the Lab's "Download this checkpoint" zip
        target = path.with_suffix("")
        if not (target / "config.json").is_file():
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
        return target

    if path.is_dir():
        if (path / "config.json").is_file():  # already a pretrained_model folder
            return path
        if (path / "pretrained_model" / "config.json").is_file():  # a checkpoints/<step> folder
            return path / "pretrained_model"
        if (path / "checkpoints").is_dir():  # a whole run folder -> newest checkpoint
            return _pick_step_dir(path / "checkpoints", None) / "pretrained_model"
        raise FileNotFoundError(f"{path} is not a checkpoint, a run folder or a pretrained_model folder")

    # Otherwise treat it as a Hugging Face Hub repo id, optionally "repo@step".
    from huggingface_hub import HfApi, snapshot_download

    repo_id, _, step = spec.partition("@")
    step = step.removeprefix("checkpoints/") or None
    files = HfApi().list_repo_files(repo_id)
    ckpt_steps = {}  # {20000: "020000"} - LeRobot zero-pads the folder names
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[0] == "checkpoints" and parts[2:] == ["pretrained_model", "config.json"]:
            ckpt_steps[int(parts[1])] = parts[1]

    if step is None and "config.json" in files:
        # Final model at the repo root (end of an HF-cloud run, or a root-layout upload).
        root_files = [f for f in files if "/" not in f and f.endswith((".json", ".safetensors"))]
        return Path(snapshot_download(repo_id, allow_patterns=root_files))
    if not ckpt_steps:
        raise FileNotFoundError(f"No LeRobot policy found in Hub repo {repo_id}")
    chosen = max(ckpt_steps) if step is None else int(step)
    if chosen not in ckpt_steps:
        raise FileNotFoundError(f"Step {step} not in {repo_id}. Available: {sorted(ckpt_steps)}")
    sub = f"checkpoints/{ckpt_steps[chosen]}/pretrained_model"
    local = snapshot_download(repo_id, allow_patterns=[f"{sub}/*"])  # weights only, no optimizer state
    return Path(local) / sub


# --------------------------------------------------------------------------------------
# Step 2 - load the policy and its pre/post-processors
# --------------------------------------------------------------------------------------
def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_policy(path: Path, device: str, act_ensemble: float | None = None):
    """Load weights + the normalisation stats saved with them. No dataset needed."""
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    # The trained backbone weights are inside model.safetensors, so skip the ImageNet download
    # that ACT/Diffusion would otherwise do on construction (handy on an offline machine).
    if hasattr(cfg, "pretrained_backbone_weights"):
        cfg.pretrained_backbone_weights = None
    if act_ensemble is not None:  # same knob as the Lab's "temporal ensembling" option (ACT only)
        cfg.temporal_ensemble_coeff = act_ensemble
        cfg.n_action_steps = 1

    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def describe_policy(cfg: PreTrainedConfig) -> dict[str, tuple[int, int]]:
    """Print what the policy expects and return {camera_name: (height, width)}."""
    cams = {}
    for key, ft in cfg.image_features.items():
        _, h, w = ft.shape
        cams[key.removeprefix("observation.images.")] = (h, w)
    print(f"Policy type : {cfg.type}")
    print(f"State dim   : {cfg.robot_state_feature.shape if cfg.robot_state_feature else None}")
    print(f"Action dim  : {cfg.action_feature.shape}")
    print(f"Cameras     : {', '.join(f'{n} ({h}x{w})' for n, (h, w) in cams.items()) or 'none'}")
    return cams


# --------------------------------------------------------------------------------------
# Step 3 - connect the SO-101 follower with the cameras the policy was trained on
# --------------------------------------------------------------------------------------
def make_robot(args, policy_cams: dict[str, tuple[int, int]]):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    given = dict(c.split("=", 1) for c in args.camera)
    missing = set(policy_cams) - set(given)
    if missing:
        raise SystemExit(
            f"The policy needs cameras {sorted(policy_cams)}; add --camera NAME=INDEX for {sorted(missing)}. "
            "Names must match the ones used when recording in MakerMods Lab."
        )
    cameras = {}
    for name, (h, w) in policy_cams.items():
        idx = given[name]
        cameras[name] = OpenCVCameraConfig(
            index_or_path=int(idx) if idx.isdigit() else idx,
            width=w, height=h, fps=args.fps,
            fourcc="MJPG",  # MakerMods Lab's default too; avoids USB bandwidth errors with 2+ cameras
        )
    cfg = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,  # must match the calibration name used in MakerMods Lab
        cameras=cameras,
        max_relative_target=args.max_relative_target,
    )
    return SO101Follower(cfg)


class FakeSO101:
    """Stand-in robot for --dry-run: same interface as SO101Follower, no hardware."""

    robot_type = "so_follower"

    def __init__(self, policy_cams: dict[str, tuple[int, int]]):
        self.cams = policy_cams
        self.observation_features = {f"{m}.pos": float for m in SO101_MOTORS}
        self.observation_features.update({n: (h, w, 3) for n, (h, w) in policy_cams.items()})
        self.action_features = {f"{m}.pos": float for m in SO101_MOTORS}
        self.sent: list[dict] = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_observation(self):
        obs = {f"{m}.pos": 0.0 for m in SO101_MOTORS}
        for n, (h, w) in self.cams.items():
            obs[n] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        return obs

    def send_action(self, action):
        self.sent.append(action)
        return action


# --------------------------------------------------------------------------------------
# Step 4 - the control loop
# --------------------------------------------------------------------------------------
def run_loop(robot, policy, preprocessor, postprocessor, device, task, fps, duration_s, max_steps=None):
    # Feature spec built from the robot itself: maps "shoulder_pan.pos"... to tensor positions.
    features = {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.observation_features, "observation"),
    }
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    use_amp = device == "cuda" and getattr(policy.config, "use_amp", False)

    step, t_start = 0, time.perf_counter()
    while time.perf_counter() - t_start < duration_s and (max_steps is None or step < max_steps):
        t0 = time.perf_counter()

        obs = robot.get_observation()                       # raw dict from motors + cameras
        frame = build_inference_frame(obs, torch.device(device), features, task=task,
                                      robot_type=robot.robot_type)
        with torch.inference_mode(), (torch.autocast(device_type="cuda") if use_amp else nullcontext()):
            batch = preprocessor(frame)                     # normalise with the training stats
            action = policy.select_action(batch)            # one action (ACT pops from its chunk)
            action = postprocessor(action)                  # un-normalise back to motor degrees
        robot.send_action(make_robot_action(action, features))  # {"shoulder_pan.pos": ..., ...}

        step += 1
        precise_sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))
    return step


def return_to_pose(robot, pose: dict[str, float], seconds: float = 3.0, fps: int = 30) -> None:
    """Glide back to the start pose so the arm doesn't drop when torque is released."""
    now = robot.get_observation()
    n = max(1, int(seconds * fps))
    for i in range(1, n + 1):
        a = i / n
        robot.send_action({k: (1 - a) * now[k] + a * pose[k] for k in pose})
        precise_sleep(1.0 / fps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list local MakerMods Lab runs + calibrations and exit")
    p.add_argument("--policy", help="run folder, checkpoint folder, .zip, or Hub repo[@step]")
    p.add_argument("--dry-run", action="store_true", help="load the policy and run 30 steps on a fake robot")
    p.add_argument("--port", help="follower serial port, e.g. /dev/ttyACM0 or /dev/tty.usbmodemXXXX or COM5")
    p.add_argument("--robot-id", help="follower calibration name from MakerMods Lab (see --list)")
    p.add_argument("--camera", action="append", default=[], help="NAME=INDEX_OR_PATH, repeat per camera")
    p.add_argument("--task", default="", help="language instruction (used by SmolVLA / pi0-style policies)")
    p.add_argument("--fps", type=int, default=30, help="control rate; match the recording fps (Lab default 30)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds to run")
    p.add_argument("--device", help="cuda / mps / cpu (auto if omitted)")
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="safety clamp: max degrees a joint may move per step (e.g. 10)")
    p.add_argument("--act-ensemble", type=float, default=None,
                   help="ACT temporal ensembling coefficient, e.g. 0.01 (smoother, slower)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        list_local_runs()
        return
    if not args.policy:
        p.error("--policy is required (or use --list)")

    device = pick_device(args.device)
    path = resolve_policy_path(args.policy)
    print(f"Checkpoint  : {path}\nDevice      : {device}")
    policy, pre, post = load_policy(path, device, args.act_ensemble)
    policy_cams = describe_policy(policy.config)

    if args.dry_run:
        robot = FakeSO101(policy_cams)
        steps = run_loop(robot, policy, pre, post, device, args.task, fps=1000, duration_s=60, max_steps=30)
        print(f"Dry run OK: {steps} steps. Last action sent: "
              + ", ".join(f"{k}={v:.2f}" for k, v in robot.sent[-1].items()))
        return

    if not (args.port and args.robot_id):
        p.error("--port and --robot-id are required to drive the real arm")
    robot = make_robot(args, policy_cams)
    robot.connect()
    obs = robot.get_observation()
    start_pose = {k: obs[k] for k in robot.action_features}  # where the arm is now (ideally rest)
    try:
        print(f"Running for {args.duration:.0f}s at {args.fps} Hz - press Ctrl+C to stop early.")
        steps = run_loop(robot, policy, pre, post, device, args.task, args.fps, args.duration)
        print(f"Finished {steps} steps.")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            print("Returning to start pose...")
            return_to_pose(robot, start_pose, fps=args.fps)
        finally:
            robot.disconnect()  # releases torque (disable_torque_on_disconnect=True)


if __name__ == "__main__":
    main()
