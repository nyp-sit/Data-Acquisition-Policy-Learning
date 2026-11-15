# Train SO101 to pick up a cube (URDF to trained policy)

[Back to topics](readme.md)

This is a complete example, from the SO101 robot description file (URDF) to
a trained policy that picks up a cube in Isaac Sim. It uses the task
`Isaac-SO-ARM101-Lift-Cube-v0`: the arm must reach a cube on a table, grasp
it, lift it, and carry it to a target position that changes every 5 seconds.

The work goes through four stages:

```text
A. Robot            B. Task                    C. Train              D. Play
URDF file      -->  scene + rewards +     -->  PPO, 1024 arms   -->  trained policy
robot config        registration               in parallel           in the Isaac Sim GUI
(Steps 1-3)         (Steps 4-8)                (Steps 9-11)          (Steps 12-13)
```

## What you need

| Item | Value |
|---|---|
| Isaac Sim / Isaac Lab Python | `C:\isaacsimlab\env_isaaclab\Scripts\python.exe` |
| SO101 project (run commands from here) | `C:\isaacsimlab\isaac_so_arm101` |
| SO101 package source | `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\` |
| Isaac Lab tools (URDF converter) | `C:\isaacsimlab\IsaacLab\scripts\tools\` |
| Training results | `C:\isaacsimlab\isaac_so_arm101\logs\rsl_rl\lift\<date_time>\` |
| Internet | Needed on the first run: the table and cube models download from NVIDIA's asset server |

Set up every PowerShell terminal you use for this guide like this:

```powershell
cd C:\isaacsimlab\isaac_so_arm101
$py = "C:\isaacsimlab\env_isaaclab\Scripts\python.exe"
```

You do **not** need to activate a Python environment. Keep the `&` in front
of `$py` in every command (`& $py ...`). Without it, PowerShell shows
`Unexpected token`.

---

## Stage A: The robot

### Step 1: Look at the SO101 URDF

The URDF describes the arm's parts (links), how they connect (joints), their
masses and their 3D meshes:

```text
C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\robots\trs_so101\
    urdf\so_arm101.urdf     robot description (XML)
    urdf\assets\*.stl       3D meshes referenced by the URDF
    so_arm101.py            Isaac Lab robot config (Step 3)
```

The SO101 URDF contains these joints and links:

| Joint (moves) | Type | Link it moves | Used in the task as |
|---|---|---|---|
| `shoulder_pan` | revolute | `shoulder_link` | arm action |
| `shoulder_lift` | revolute | `upper_arm_link` | arm action |
| `elbow_flex` | revolute | `lower_arm_link` | arm action |
| `wrist_flex` | revolute | `wrist_link` | arm action |
| `wrist_roll` | revolute | `gripper_link` | arm action; `gripper_link` is the end-effector body |
| `gripper` | revolute | `moving_jaw_so101_v1_link` | gripper action (open/close) |
| `gripper_frame_joint` | fixed | `gripper_frame_link` | not used |

The fixed root link is `base_link`. **The joint and link names matter.** The
robot config and the task refer to them by name (Steps 3 and 5). If you use
a different URDF, these names must match.

To list them yourself:

```powershell
Select-String -Path src\isaac_so_arm101\robots\trs_so101\urdf\so_arm101.urdf -Pattern '<joint name=|<link name='
```

### Step 2 (optional): Convert the URDF to USD and view it in Isaac Sim

Isaac Sim works with USD files. **You don't need to do this step for
training.** Isaac Lab converts the URDF to USD automatically every time a
task starts (Step 3). Do it when you want to look at the robot in the GUI
and check that its meshes, joints and orientation are correct.

```powershell
& $py C:\isaacsimlab\IsaacLab\scripts\tools\convert_urdf.py `
    src\isaac_so_arm101\robots\trs_so101\urdf\so_arm101.urdf `
    C:\isaacsimlab\so101_usd\so_arm101.usd `
    --fix-base --joint-stiffness 100 --joint-damping 1
```

- Isaac Sim opens, prints `Generated USD file: C:\isaacsimlab\so101_usd\so_arm101.usd`
  and shows the arm. Close the window when you're done.
- To convert only, without opening the GUI, add `--headless`.
- `--fix-base` bolts the base to the ground. Without it, the arm falls over
  when you press Play.
- To inspect it in the GUI, press **Play** (the arm should stand still and
  hold its pose). Then find a joint such as `shoulder_lift` in the **Stage**
  panel (use its search box) and change its drive target in the
  **Property** panel to see it move.
- The stiffness and damping set here only affect this USD file. Training
  uses the values from Step 3.

### Step 3: The robot config (`SO_ARM101_CFG`)

File: `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\robots\trs_so101\so_arm101.py`

This turns the URDF into a robot that Isaac Lab can simulate and control:

| Setting | Value | Meaning |
|---|---|---|
| `spawn=UrdfFileCfg(asset_path=.../so_arm101.urdf)` | the URDF from Step 1 | Converted to USD automatically at start-up |
| `fix_base=True` | | Base is fixed to the table |
| `replace_cylinders_with_capsules=True` | | Simpler collision shapes, more stable physics |
| `enabled_self_collisions=True` | | The arm can't pass through itself |
| `init_state.joint_pos` | all `0.0`, except `wrist_flex: 1.57` | Starting pose at each reset |
| `actuators["arm"]` | stiffness 200 / 170 / 120 / 80 / 50, damping 80 / 65 / 45 / 30 / 20 | Position-controlled motors for pan, lift, elbow, wrist flex, wrist roll. Stiffer for joints that carry more mass |
| `actuators["arm"]` limits | effort 1.9, velocity 1.5 | Motor torque and speed limits |
| `actuators["gripper"]` | stiffness 60, damping 20, effort 2.5 | Gripper motor, made stronger for a reliable grip |
| `soft_joint_pos_limit_factor=0.9` | | Use 90 % of each joint's URDF range |

**Using your own URDF** (for example, a modified SO101): copy the
`trs_so101` folder, change `asset_path` to your URDF, and update the
joint names in `init_state` and `actuators` to match your URDF. Then use
your new config in Step 5 in place of `SO_ARM101_CFG`.

---

## Stage B: The task

A task (environment) is Python configuration, not a saved Isaac Sim scene.
The training script builds the scene from it when it starts.

### Step 4: The lift environment (`LiftEnvCfg`)

File: `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\tasks\lift\lift_env_cfg.py`. This describes the task in general,
for any robot:

| Part | What it contains |
|---|---|
| **Scene** (`ObjectTableSceneCfg`) | A table (`SeattleLabTable`), a ground plane, a dome light, plus slots for the robot, the cube and an end-effector frame sensor (filled in Step 5) |
| **Command** (`object_pose`) | A random target position for the cube: x -0.1..0.1 m, y -0.3..-0.1 m, z 0.2..0.35 m (relative to the robot base). A new target every 5 s. Shown as a marker in the GUI |
| **Observations** (what the policy sees) | Joint positions and velocities, the cube position in the robot frame, the target position, the last action |
| **Actions** | Slots for the arm action and the gripper action (filled in Step 5) |
| **Events** | At each reset: robot back to its start pose, and the cube placed randomly within ±0.1 m (x) and ±0.2 m (y) |
| **Terminations** | Time-out after 5 s, or the cube falls off the table (height below -0.05 m) |
| **Curriculum** | After 10,000 steps, the action-rate and joint-velocity penalties increase from -0.0001 to -0.1, to make motion smoother |
| **Simulation** | Physics at 100 Hz (`dt = 0.01`). The policy acts every 2 physics steps (50 Hz) |

**Rewards.** These teach the robot, in order of what it learns first:

| Reward term | Weight | Given when |
|---|---|---|
| `reaching_object` | 1.0 | The gripper is close to the cube |
| `lifting_object` | 15.0 | The cube is higher than 0.025 m |
| `object_goal_tracking` | 16.0 | The lifted cube is near the target (coarse, std 0.3 m) |
| `object_goal_tracking_fine_grained` | 5.0 | The lifted cube is very near the target (fine, std 0.05 m) |
| `action_rate` | -0.0001 (later -0.1) | Penalty for jerky action changes |
| `joint_vel` | -0.0001 (later -0.1) | Penalty for fast joint motion |

### Step 5: Put the SO101 into the lift task (`SoArm101LiftCubeEnvCfg`)

File: `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\tasks\lift\joint_pos_env_cfg.py`. This fills in the robot-specific
slots of Step 4, using the URDF names from Step 1:

| Setting | Value |
|---|---|
| `scene.robot` | `SO_ARM101_CFG` (Step 3) |
| `actions.arm_action` | `JointPositionActionCfg` on `shoulder_.*`, `elbow_flex`, `wrist_.*` (5 joints). Target = start pose + 0.5 × action |
| `actions.gripper_action` | `BinaryJointPositionActionCfg` on `gripper`: open = 0.5 rad, close = 0.0 rad |
| `commands.object_pose.body_name` | `gripper_link` |
| `scene.object` | A cube (NVIDIA `DexCube`, scaled to 0.5) at (0.2, 0, 0.015) m in front of the robot |
| `scene.ee_frame` | Frame sensor from `base_link` to `gripper_link`, offset (0.01, 0, -0.09) m, which puts it between the gripper jaws. The reward uses it to measure the gripper-to-cube distance |

`SoArm101LiftCubeEnvCfg_PLAY` is the same task with 50 environments and no
observation noise. It's used for playback in Step 12.

### Step 6: Register the task name

File: `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\tasks\lift\__init__.py`

```python
gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101LiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
```

This connects the name you pass to `--task` with the environment config
(Step 5) and the training config (Step 7). The same file registers
`Isaac-SO-ARM101-Lift-Cube-Play-v0`. Check that both are visible:

```powershell
& $py -m isaac_so_arm101.scripts.list_envs
```

### Step 7: The training config (`LiftCubePPORunnerCfg`)

File: `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\tasks\lift\agents\rsl_rl_ppo_cfg.py`

| Setting | Value | Meaning |
|---|---|---|
| `max_iterations` | 1500 | Training length |
| `num_steps_per_env` | 24 | Steps each arm collects per iteration |
| `save_interval` | 50 | Save a checkpoint every 50 iterations |
| `experiment_name` | `lift` | Results go to `logs\rsl_rl\lift\` |
| `actor_hidden_dims` / `critic_hidden_dims` | [256, 128, 64] | Neural network size |
| `learning_rate` | 1e-4 (adaptive) | How fast the network changes |
| `entropy_coef` | 0.006 | Encourages exploration |
| `gamma` | 0.98 | How much future rewards count |

### Step 8: Preview the task in the GUI (before training)

Check that the scene looks right (arm, table, cube, target marker) before
spending time on training.

Arms that hold still (zero actions):

```powershell
& $py -m isaac_so_arm101.scripts.zero_agent --task Isaac-SO-ARM101-Lift-Cube-v0 --num_envs 4
```

Arms that move randomly (random actions; shows that the arm and gripper
respond):

```powershell
& $py -m isaac_so_arm101.scripts.random_agent --task Isaac-SO-ARM101-Lift-Cube-v0 --num_envs 4
```

- The first start can take a few minutes (shader compiling, and downloading
  the table and cube models).
- You should see 4 copies of: an SO101 on a table, a small cube in front of
  it, a frame marker between the gripper jaws, and a target marker in the air.
- Close the window when you're done. **Close it before training**, because
  the GPU has only 8 GB.

To drive the arm yourself in this scene, use the live sim from
[Test SO101 arm movement](test-so101-arm-movement.md) with this task:
`.\tools\launch_live_sim.ps1 -Task Isaac-SO-ARM101-Lift-Cube-v0 -NumEnvs 2`
(run from `C:\SO101wClaudeIsaac`).

---

## Stage C: Train

### Step 9: Run a short smoke test

```powershell
& $py -m isaac_so_arm101.scripts.rsl_rl.train --task Isaac-SO-ARM101-Lift-Cube-v0 --headless --num_envs 64 --max_iterations 10
```

It succeeds when you see `Learning iteration 9/10` and the script exits
without an error. This creates a small run in `logs\rsl_rl\lift\`, which you
can delete.

### Step 10: Run the full training

```powershell
& $py -m isaac_so_arm101.scripts.rsl_rl.train --task Isaac-SO-ARM101-Lift-Cube-v0 --headless --num_envs 1024
```

- Runs 1500 iterations. Lift is slower than Reach (bigger network, cube
  contact physics). Expect **tens of minutes** (not yet measured on this
  laptop).
- Use `--num_envs 1024` or lower. If you get `CUDA out of memory`, use `512`.
- Always use `--headless` for training.
- If the cube isn't lifted well by the end, train longer: add
  `--max_iterations 3000`, or continue the run (Step 11).

### Step 11: Watch progress, and continue if needed

In a second terminal:

```powershell
& "C:\isaacsimlab\env_isaaclab\Scripts\python.exe" C:\SO101wClaudeIsaac\tools\tb_summary.py --filter Episode_Reward
```

Or view graphs in TensorBoard (then open <http://localhost:6006>):

```powershell
& $py -m tensorboard.main --logdir logs\rsl_rl\lift
```

The rewards should go up in this order, which follows the steps of the task:

| Metric | What it tells you |
|---|---|
| `Episode_Reward/reaching_object` | Rises first: the gripper learns to go to the cube |
| `Episode_Reward/lifting_object` | Rises next: the cube gets lifted. If this stays near 0, the robot hasn't learned to grasp yet |
| `Episode_Reward/object_goal_tracking` (and `_fine_grained`) | Rises last: the lifted cube is carried to the target |
| `Train/mean_reward` | Overall score. Higher is better |
| `Episode_Termination/object_dropping` | Share of episodes where the cube fell off the table. Should stay low |

To continue training from the last checkpoint, into a new run folder:

```powershell
& $py -m isaac_so_arm101.scripts.rsl_rl.train --task Isaac-SO-ARM101-Lift-Cube-v0 --headless --num_envs 1024 --resume --load_run <run_folder> --checkpoint model_1499.pt --max_iterations 1000
```

Replace `<run_folder>` with the folder name under `logs\rsl_rl\lift\`, for
example `2026-10-05_21-00-00`.

---

## Stage D: Play the trained policy in Isaac Sim

### Step 12: Watch the trained arm pick up the cube

```powershell
& $py -m isaac_so_arm101.scripts.rsl_rl.play --task Isaac-SO-ARM101-Lift-Cube-Play-v0 --num_envs 4
```

- Use the **`-Play-v0`** task (no observation noise).
- It loads the **last checkpoint of the most recent run** in
  `logs\rsl_rl\lift\` automatically.
- Each arm reaches for its cube, closes the gripper, lifts the cube and moves
  it toward the target marker. The target changes every 5 s, and the scene
  resets every 5 s.
- Close the window to stop.

To play a specific run or checkpoint, for example to compare an early and a
late checkpoint:

```powershell
& $py -m isaac_so_arm101.scripts.rsl_rl.play --task Isaac-SO-ARM101-Lift-Cube-Play-v0 --num_envs 4 --load_run <run_folder> --checkpoint model_500.pt
```

### Step 13: Save a video and the exported policy

```powershell
& $py -m isaac_so_arm101.scripts.rsl_rl.play --task Isaac-SO-ARM101-Lift-Cube-Play-v0 --num_envs 4 --headless --video --video_length 500
```

After any play run, the run folder contains:

```text
logs\rsl_rl\lift\<run_folder>\
    model_0.pt ... model_1499.pt   checkpoints
    params\env.yaml, agent.yaml    settings used for this run
    exported\policy.pt             TorchScript policy (for use outside Isaac Lab)
    exported\policy.onnx           ONNX policy
    videos\play\*.mp4              video (with --video)
```

---

## Make your own variant

To change the task without losing the original, for example a bigger cube,
a different target range or new reward weights:

1. Check `git status` in `C:\isaacsimlab\isaac_so_arm101`.
2. In `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\tasks\lift\joint_pos_env_cfg.py`, add a class that subclasses
   `SoArm101LiftCubeEnvCfg` and change what you need in `__post_init__`:

   ```python
   @configclass
   class SoArm101LiftCubeEasyEnvCfg(SoArm101LiftCubeEnvCfg):
       def __post_init__(self):
           super().__post_init__()
           self.commands.object_pose.ranges.pos_z = (0.1, 0.2)   # lower targets
           self.rewards.lifting_object.weight = 20.0
   ```

3. Register it with a new name in `C:\isaacsimlab\isaac_so_arm101\src\isaac_so_arm101\tasks\lift\__init__.py`, as in
   Step 6 (for example `Isaac-SO-ARM101-Lift-Cube-Easy-v0`).
4. Run Steps 9–12 with the new name, then commit the change.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Unexpected token` | The `&` before `$py` is missing. |
| Task not found (`NameNotFound`) | Include the `Isaac-` prefix: `Isaac-SO-ARM101-Lift-Cube-v0`. |
| No output for several minutes on first start | Shaders are compiling and models are downloading. Wait 5–10 minutes. |
| Error loading `table_instanceable.usd` or `dex_cube_instanceable.usd` | No internet access to NVIDIA's asset server. Check the connection and try again. |
| `CUDA out of memory` | Close other Isaac Sim windows (preview, play, live sim) and lower `--num_envs`. |
| Arm falls over in the Step 2 viewer | Convert again with `--fix-base`. |
| `lifting_object` reward stays near 0 | The robot hasn't learned to grasp. Train longer (Step 11). Then check the gripper open/close values and the `ee_frame` offset (Step 5). |
| Play loads the wrong run | Use `--load_run <run_folder> --checkpoint <file>`. |
| Results not in `C:\isaacsimlab\isaac_so_arm101\logs` | The command was run from another folder. `cd` there first. |
