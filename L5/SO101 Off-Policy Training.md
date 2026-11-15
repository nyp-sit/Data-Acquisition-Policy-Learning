Lab 4 · Supplement C · Beginner guide · Real hardware

# Off-policy RL on the real SO101

Appendix D of Lab 4 says LeRobot trains the real SO101 with **SAC**, an **off-policy** algorithm, inside a method called **HIL-SERL** (human-in-the-loop, sample-efficient RL). This guide starts from zero: what off-policy means, how it is used to teach the arm to lift a cube, exactly which rewards and settings are involved, and the step-by-step procedure to collect the data and run the training.

**LeRobot version:** 0.6.2 (source checked at commit `45cca1a`) **Task:** pick and lift a cube with one SO101 follower arm **Builds on:** Lab 4 Appendix D and Supplement B (real-robot rewards)

[1 · ConceptWhat off-policy training isOn-policy vs off-policy, replay buffer, SAC in plain words](#offpolicy) [2 · ApplicationHow it trains the SO101Actor, learner, two buffers, human take-overs](#applied) [3 · Rewards & settingsWhat the arm is rewarded forSuccess reward, gripper penalty, every parameter](#reward) [4 · ProcedureCollect data and train10 steps from calibration to a trained policy](#procedure)

1 · Concept

## What "off-policy" training means

Every RL algorithm repeats the same cycle: the **policy** (the robot's decision maker) acts, the robot records what happened, and the algorithm updates the policy so that good outcomes become more likely. The difference between on-policy and off-policy is **which recorded experience the algorithm is allowed to learn from**.

On-policy · PPO in Lab 4

Learns only from experience produced by the **current** version of the policy. After each update, that experience is out of date and is thrown away. Like a student who may only study the homework they did today, then must bin it.

Off-policy · SAC on the real arm

Learns from experience produced by **any** policy: older versions of itself, random exploration, or a **human operating the arm**. Everything is kept in a **replay buffer** and reused many times. Like a student who keeps every worked example, including the teacher's, and revises them again and again.

**Figure 1.** The off-policy loop has a memory. Because the update does not care who produced a step, the human's demonstrations and corrections can be learned from in the same way as the robot's own attempts.

### Why the real arm needs off-policy learning

Isaac Lab, Lab 4 run51,200 /s

steps of experience per second: 1024 arms × 50 Hz. The full run used 1500 × 24 × 1024 ≈ **36.9 million** steps.

One real SO101≤ 28,800 /h

steps per **hour**: 10 Hz, 20 s episodes plus a 5 s reset. It would take **1,280 hours** (about 53 days, non-stop) to match the sim run.

What HIL-SERL does insteada few hours

Reuses every step many times, starts from human demonstrations, and lets a human correct the arm. The LeRobot guide targets training "in just a few hours".

### Key words, in plain language

| Term | Meaning | In the SO101 lift task |
| --- | --- | --- |
| **Policy** (actor) | The network that chooses the action from what the robot sees. | Sees 2 camera images + joint readings; outputs a small gripper move (Δx, Δy, Δz) and open/stay/close. |
| **Transition** | One recorded step: state, action, reward, next state, done. | One 0.1 s step of the arm (10 per second). |
| **Replay buffer** | The memory of past transitions the learner samples from. | Two buffers: **online** (everything) and **offline** (human data only). |
| **Critic** (Q-function) | A second network that predicts "how much reward will I eventually get if I take this action here?" | Learns that "jaws closed around the cube, moving up" is worth almost 1.0. |
| **Discount** γ | How much a future reward is worth now. 0.99 means each step of delay costs 1 %. | A success after 50 steps is worth 0.99⁵⁰ ≈ 0.61 at the start, so faster lifts score higher. |
| **Temperature** α | How strongly SAC rewards trying varied actions (exploration). | Start low (0.01) so that the human's corrections are not drowned out by random motion. |
| **Sparse reward** | Reward only at the very end, when the goal is reached. | 0 on every step, 1 when the cube is lifted. |

### SAC in three sentences

SAC (Soft Actor-Critic) trains two kinds of network at once. The **critic** learns to predict the future reward of an action, using the rule "value now = reward now + 0.99 × value of the next step". The **actor** (the policy) is then pushed toward actions the critic scores highly, while a small bonus for randomness keeps it exploring.

Critic target y = r + 0.99 × (1 − done) × \[ min(Q₁, Q₂)(s′, a′) − α · log π(a′|s′) \] Actor goal choose a to maximise Q(s, a) − α · log π(a|s) Gripper a separate "discrete" critic scores {close, stay, open}; its target adds the gripper penalty

Two critics (Q₁, Q₂) are kept and the smaller is used, so the robot does not chase an over-optimistic guess. Slow-moving "target" copies of the critics (updated by 0.5 % each step) keep the targets stable.

2 · Application

## How HIL-SERL uses off-policy SAC to teach the lift

SAC alone, with a reward that is 0 almost all the time, would wander for a very long time before it ever lifted the cube by chance. HIL-SERL adds four ingredients so the arm finds success quickly and safely:

1 · Demonstrations

You first drive the arm with a gamepad 15 to 20 times to grasp and lift the cube. These episodes fill the **offline buffer** before training starts.

2 · Human take-overs

During training you hold a button to take control whenever the arm goes wrong. Those corrected steps are saved to **both** buffers.

3 · Success reward

Reward 1 when the cube is lifted, judged by your button press or by a trained camera classifier. 0 otherwise.

4 · Small, safe action space

The policy moves the gripper by at most 2 cm per step inside a small box above the table. Inverse kinematics converts that into joint angles.

**Figure 2.** The actor keeps the robot moving in real time while the learner trains on the GPU, so learning never pauses the arm. The offline buffer is what makes human data count: half of every batch comes from it, so the arm keeps re-studying how a person did the lift.

### What one transition contains for the lift task

**state s**front image 128×128 · wrist image 128×128 · 6 joint positions (degrees, gripper 0–100) · optional joint velocities

**action a**Δx, Δy, Δz ∈ \[−1, 1\], each × 0.02 m · gripper 0 close / 1 stay / 2 open

**reward r**1.0 if the cube is lifted this step, otherwise 0.0

**next state s′**the same readings 0.1 s later

**done**true on success, failure button, or step 200 (20 s)

**extra info**is_intervention (did you drive this step?) · discrete_penalty (gripper)

### One training episode, step by step

**Figure 3.** An example episode (numbers illustrative, rules exact). The policy struggled, you steered it to the cube and closed the gripper, then it finished the lift. The learner gets 93 new transitions; the 27 you produced are also kept as extra "demonstration" data.

### How a training session usually develops

Early

#### You do a lot of the work

your share of controlhigh

Let it explore a few episodes first, then step in whenever it drifts away from the cube. Most successes still need your help.

Middle

#### Short corrections only

your share of controlmedium

The arm reaches the cube on its own. You give brief nudges, often just the grasp. Avoid long take-overs.

Late

#### The arm lifts on its own

your share of controllow

The intervention rate in Weights & Biases falls toward zero while the episodic reward stays near 1. Time to evaluate.

The meters show the typical direction of change described in the LeRobot guide, not measured values.

3 · Rewards & settings

## What the SO101 is rewarded for, and every setting that matters

Unlike the six shaped terms in Isaac Lab, the real arm has one main reward. The robot hardware itself never produces a reward: `RobotEnv.step()` returns 0.0, and processor steps add the reward afterwards.

reward at step t = 1.0 if the cube is judged lifted at step t (episode ends) = 0.0 otherwise gripper critic only: + discrete_penalty (−0.02 when the command swings the gripper fully open↔closed, if enabled)

| Who decides | How | Value | Setting |
| --- | --- | --- | --- |
| You | Press success (gamepad Y, or `s` on the keyboard teleop) when the cube is clearly off the table | 1.0, done | `reset.terminate_on_success: true` |
| Camera classifier optional | A small image model trained on your recorded episodes predicts "lifted" from the two cropped images | 1.0 if p > threshold | `reward_classifier.pretrained_path`, `success_threshold` 0.5 (guide example 0.7) |
| You (fail) | Press failure (gamepad A) if the cube falls or the arm gets stuck | 0.0, done | — |
| Clock | No success after `control_time_s × fps` = 20 × 10 = 200 steps | 0.0, done | `reset.control_time_s` |
| Gripper encoder optional | Gripper reads open (\< 10 of 100) and is commanded closed, or the reverse | −0.02 | `gripper.gripper_penalty` (default 0.0 = off) |

### How a single +1 teaches every earlier step

With only one reward per episode, the critic spreads it backwards using the discount. For the episode in Figure 3 (success at step 92), once the critic has learned, its targets look like this (exploration bonus left out to keep the numbers simple):

**Figure 4.** Bars are drawn to scale (0.9992 − k). Step 91 has reward 0 but target 0 + 0.99 × 1.0 = 0.99, step 90 gets 0.99 × 0.99 = 0.98, and so on. A failed episode gives every step a target near 0, so the critic learns which situations lead to a lift and which do not.

### All the parameters for the lift task

Values are LeRobot 0.6.2 defaults unless marked "guide" (the LeRobot HIL-SERL guide's example) or "set" (you must choose it). Section 4 shows where each one goes in the config files.

| Parameter | Config key | Value for the lift | What it does |
| --- | --- | --- | --- |
| Episode timing · env_config.json → env |  |  |  |
| Control rate | `fps` | 10 Hz (guide) | One policy step every 0.1 s. |
| Episode length | `processor.reset.control_time_s` | 20 s → 200 steps | Time limit; ends with reward 0. A lift should take 5–10 s. |
| Reset wait | `processor.reset.reset_time_s` | 5 s | Time to put the cube back after the arm returns home. |
| Home pose | `processor.reset.fixed_reset_joint_positions` | set (6 values, degrees; gripper 0–100) | Where every episode starts. Pick a pose above the workspace with the gripper open. |
| Stop on success | `processor.reset.terminate_on_success` | true | Set false only when recording data for the classifier. |
| Reward |  |  |  |
| Success value | `processor.reward_classifier.success_reward` | 1.0 | Reward on the success step. |
| Classifier model | `processor.reward_classifier.pretrained_path` | optional | Leave out to label success yourself with the gamepad. |
| Classifier threshold | `processor.reward_classifier.success_threshold` | 0.5 (guide 0.7) | Higher = fewer false "lifted" rewards. |
| Gripper penalty | `processor.gripper.gripper_penalty` | 0.0 → try −0.02 | Discourages pointless open/close switching. |
| Action & safety · the hardware limits |  |  |  |
| Arm model | `processor.inverse_kinematics.urdf_path`, `target_frame_name` | `so101_new_calib.urdf`, `gripper` | Forward / inverse kinematics between joints and gripper position. |
| Workspace box | `processor.inverse_kinematics.end_effector_bounds` | set, e.g. min \[0.16, −0.08, 0.03\] max \[0.24, 0.20, 0.10\] m | Gripper targets are clipped into this box. Small box = faster learning. |
| Step size | `processor.inverse_kinematics.end_effector_step_sizes` | x, y, z 0.02 m (guide) | Action 1.0 moves the gripper 2 cm in one step. |
| Jump limit | `EEBoundsAndSafety.max_ee_step_m` (in code) | 0.05 m | A bigger jump in one step stops the loop with an error. |
| Gripper | `processor.gripper.use_gripper`, `processor.max_gripper_pos` | true, 100 | Adds the open/stay/close action; gripper range 0–100. |
| Robot hardware · env_config.json → env.robot |  |  |  |
| Robot type | `robot.type`, `port`, `id` | `so101_follower`, set, set | `id` selects your calibration file. |
| Servo controller | `position_p/i/d_coefficient` | 16 / 0 / 32 | How stiffly the STS3215 servos track targets (real counterpart of Isaac K and D). |
| Gripper strength | written at connect | torque limit 50 % | Protects the gripper servo; enough to hold a light cube. |
| Cameras | `robot.cameras` | front + wrist (set) | Images for the policy and the classifier. |
| Observation |  |  |  |
| Image crop | `processor.image_preprocessing.crop_params_dict` | from the crop tool (Step 6) | Removes background that would confuse learning. |
| Image size | `processor.image_preprocessing.resize_size` | \[128, 128\] | 64×64 also works if GPU memory is short. |
| Joint velocity | `processor.observation.add_joint_velocity_to_observation` | optional | (angle now − angle before) ÷ 0.1 s. |
| SAC learning · train_config.json → algorithm / policy |  |  |  |
| Algorithm | `algorithm.type`, `policy.type` | `sac`, `gaussian_actor` | SAC with a Gaussian (continuous) actor. |
| Discount | `algorithm.discount` | 0.99 | Figure 4. |
| Exploration | `algorithm.temperature_init` | 1.0 → 0.01 (guide) | Too high makes your take-overs ineffective. |
| Learning rates | `actor_lr`, `critic_lr`, `temperature_lr` | 3e-4 each | Size of each network update. |
| Critics | `num_critics`, `critic_target_update_weight` | 2, 0.005 | Two critics, slow target copies. |
| Gripper actions | `policy.num_discrete_actions` | 3 | close / stay / open. |
| Image encoder | `policy.vision_encoder_name`, `freeze_vision_encoder` | `lerobot/resnet10`, true | Pretrained, frozen feature extractor for the images. |
| Replay buffers & actor-learner |  |  |  |
| Buffer sizes | `policy.online_buffer_capacity`, `offline_buffer_capacity` | 100,000 each | About 2.8 h of robot steps each. |
| Warm-up | `policy.online_step_before_learning` | 100 steps | Learner waits for 100 online transitions (10 s) before updating. |
| Batch mix | `online_ratio` | 0.5 | Half of each batch from human data. |
| Weight push | `policy.actor_learner_config.policy_parameters_push_frequency` | 4 s (guide: 1–2 s) | How fresh the actor's copy of the policy is. |
| Training limit | `policy.online_steps` | 1,000,000 | Upper bound; you normally stop much earlier. |

4 · Procedure

## Collecting the data and training, step by step

The work splits into five phases. Each phase produces something the next one needs.

**Figure 5.** The whole workflow. Most of your hands-on time goes into Collect and Train, where you are the teacher.

1

Set up

#### Install LeRobot with HIL-SERL and check the hardware

```
pip install -e ".[hilserl]"
git clone https://github.com/TheRobotStudio/SO-ARM100.git   # URDF: SO-ARM100/Simulation/SO101/so101_new_calib.urdf
lerobot-find-port        # USB port of the follower (and leader)
lerobot-find-cameras opencv   # index of the front and wrist cameras
```

You need: the SO101 follower, a **gamepad** (recommended by the guide), a front camera and a wrist camera, an NVIDIA GPU, a 2–3 cm cube, and the SO101 leader arm for Step 3.

2

Set up

#### Calibrate the arm

```
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```

**Check:** the joint readings must match the URDF's zero pose, or the kinematics put the gripper in the wrong place and the workspace box is wrong.

3

Set up

#### Find the workspace box for the lift

```
lerobot-find-joint-limits \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --urdf_path=SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name=gripper --teleop_time_s=30
```

For 30 s, move the leader so the gripper visits every place the cube can be, down to grasp height and up to lift height (a few cm above the table). The script prints `max_ee` and `min_ee`: copy them into `end_effector_bounds`.

**Check:** keep the box small. Every centimetre you add is space the policy has to explore with a reward that is almost always 0.

4

Collect

#### Get the example configs and fill in env_config.json

```
hf download lerobot/config_examples rl/env_config.json rl/train_config.json --repo-type dataset --local-dir .
```

Edit `rl/env_config.json`. The block below shows the settings for an SO101 lift in recording mode; merge them into the downloaded file. Replace every port, id, camera index, bound and pose with your own.

```
{
  "env": {
    "type": "gym_manipulator",
    "name": "real_robot",
    "fps": 10,
    "robot": {
      "type": "so101_follower",
      "port": "/dev/ttyACM0",
      "id": "my_follower",
      "cameras": {
        "front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
      }
    },
    "teleop": {"type": "gamepad", "use_gripper": true},
    "processor": {
      "control_mode": "gamepad",
      "observation": {"add_joint_velocity_to_observation": true, "display_cameras": false},
      "image_preprocessing": {"crop_params_dict": {}, "resize_size": [128, 128]},
      "gripper": {"use_gripper": true, "gripper_penalty": -0.02},
      "reset": {
        "fixed_reset_joint_positions": [0.0, -20.0, 20.0, 70.0, 0.0, 40.0],
        "reset_time_s": 5.0,
        "control_time_s": 20.0,
        "terminate_on_success": true
      },
      "inverse_kinematics": {
        "urdf_path": "SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        "target_frame_name": "gripper",
        "end_effector_bounds": {"min": [0.16, -0.08, 0.03], "max": [0.24, 0.20, 0.10]},
        "end_effector_step_sizes": {"x": 0.02, "y": 0.02, "z": 0.02}
      },
      "max_gripper_pos": 100
    }
  },
  "dataset": {
    "repo_id": "<hf_user>/so101_lift_cube",
    "root": null,
    "task": "pick and lift the cube",
    "num_episodes_to_record": 20,
    "replay_episode": null,
    "push_to_hub": true
  },
  "mode": "record",
  "device": "cuda"
}
```

The home pose numbers are placeholders: move the arm to a safe pose above the workspace and read its joint values (degrees; gripper 0–100). Camera keys `front` and `wrist` become `observation.images.front` and `observation.images.wrist`.

5

Collect

#### Record 15–20 demonstrations

```
python -m lerobot.rl.gym_manipulator --config_path rl/env_config.json
```

1. The arm moves to the home pose; place the cube at a new spot inside the box during the 5 s reset.
2. **Hold RB** (the intervention button) the whole time you drive: without it the arm stays still. Use the sticks to bring the gripper over the cube and lower it, then press the gripper-close button (see the gamepad map image in the LeRobot guide; button indices vary by controller).
3. Lift the cube a few centimetres, then press **Y** (success). The episode is saved with reward 1 on that step.
4. Made a mess? Press **X** (re-record) to discard it. Press **A** to end an episode as a failure.

Do

- Vary the cube position across the whole box.
- Move smoothly and directly; finish in 5–10 s.
- Lift clearly, so success is visible in both cameras.
- Keep lighting and background the same as during training.

Avoid

- Hesitating, re-grasping or opening and closing repeatedly.
- Pressing success before the cube is off the table.
- People or moving objects in the camera view.
- Cube positions outside the workspace box.

**Data budget:** 20 episodes × about 8 s × 10 steps/s ≈ 1,600 transitions. These pre-fill the offline buffer.

6

Prepare

#### Crop the camera images to the workspace

```
python -m lerobot.rl.crop_dataset_roi --repo-id <hf_user>/so101_lift_cube
```

For each camera, draw a rectangle around the area where the gripper and cube move, then press `c`. The tool prints `(top, left, height, width)` for each camera and writes a cropped copy, `<hf_user>/so101_lift_cube_cropped_resized`. Paste the numbers into `crop_params_dict` in both config files.

7

Prepare · optional

#### Train a reward classifier (so you no longer press success)

1. Record about 20 more episodes with `"terminate_on_success": false`: after pressing success, keep the cube in the air for a few seconds so there are many "lifted" frames.
2. Train: `lerobot-train --config_path reward_classifier_train_config.json` (reward model type `reward_classifier`, ResNet-10 encoder, 2 cameras, 128×128; example config in the guide).
3. Test: set `reward_classifier.pretrained_path` and `success_threshold: 0.7`, set `"mode": null`, run `gym_manipulator` again and check it fires only when the cube is really lifted.

Skip this step for your first runs. Pressing success yourself is simpler and always correct.

8

Train

#### Fill in train_config.json

Start from the downloaded `rl/train_config.json`. Copy your whole `env` block into it (with the crop numbers), point `dataset` at the cropped demos, and check these keys:

```
{
  "dataset": {"repo_id": "<hf_user>/so101_lift_cube_cropped_resized"},
  "env":     { ... same env block as env_config.json, crop_params_dict filled ... },
  "policy": {
    "type": "gaussian_actor",
    "device": "cuda",
    "storage_device": "cuda",
    "vision_encoder_name": "lerobot/resnet10",
    "freeze_vision_encoder": true,
    "num_discrete_actions": 3,
    "online_step_before_learning": 100,
    "online_buffer_capacity": 100000,
    "offline_buffer_capacity": 100000,
    "actor_learner_config": {"learner_host": "127.0.0.1", "learner_port": 50051,
                             "policy_parameters_push_frequency": 2}
  },
  "algorithm": {"type": "sac", "discount": 0.99, "temperature_init": 0.01},
  "online_ratio": 0.5,
  "wandb": {"enable": true, "project": "so101-hilserl-lift"}
}
```

**Check:** the policy's `input_features` / `output_features` in the example must match your cameras and state size (6 positions, or 12 with joint velocities).

9

Train

#### Start the learner, then the actor, and teach

```
# terminal 1 (GPU)
python -m lerobot.rl.learner --config_path rl/train_config.json
# terminal 2 (same PC or the one wired to the arm)
python -m lerobot.rl.actor --config_path rl/train_config.json
```

| Situation | What you do | Gamepad |
| --- | --- | --- |
| First few episodes | Watch; let it explore | — |
| Arm drifts away or will clearly miss | Take over briefly, steer to the cube, release | hold **RB** |
| Cube clearly lifted (no classifier) | Mark success | **Y** |
| Cube dropped / arm stuck | End as failure | **A** |
| Arm reaches the cube reliably | Only help with the grasp, then less and less | short **RB** |

**Watch in Weights & Biases:** "Intervention rate" should fall over time and "Episodic reward" should rise toward 1. If the intervention rate stays high, see the troubleshooting table.

10

Evaluate

#### Test hands-off and keep the best checkpoint

Run the actor again without intervening for at least 10 episodes, placing the cube at different spots. Count successes (episodic reward 1) and note the time to lift. Checkpoints are saved in the learner's output folder every `save_freq` steps.

LeRobot also contains `lerobot.rl.eval_policy`, but in the version checked it does not run the reward processors, so its reward is always 0 on the real arm. Counting hands-off successes yourself is more reliable.

### Troubleshooting

| What you see | Likely cause | What to try |
| --- | --- | --- |
| Learner prints nothing about updates | Fewer than 100 online transitions yet | Wait about 10 s of robot time; check the actor is connected (port 50051). |
| `TypeError ... get_teleop_events()` | Teleoperator without event support | Use `gamepad` or the keyboard end-effector teleop. |
| `ValueError: EE jump ... > 0.05m` | Calibration or kinematics mismatch makes the target jump | Recalibrate; check `urdf_path` and `target_frame_name`. |
| Arm presses against the box edges | Box too small or in the wrong place | Redo Step 3 and include the full lift height. |
| Never grasps; intervention rate stays high | Exploration too wild or box too big | `temperature_init` 0.01, shrink the box, more demonstrations. |
| Gripper opens and closes constantly | No cost for switching | Set `gripper_penalty` to −0.02. |
| "Success" while the cube is on the table | Classifier false positive | Raise the threshold to 0.7–0.8, check the crop, record more failure frames. |

Sources and omissions

## Where this comes from

- [docs/source/hilserl.mdx](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/docs/source/hilserl.mdx): workflow, commands, example values (fps 10, 0.02 m steps, bounds, 128×128, threshold 0.7, temperature 0.01, intervention advice).
- [rl/learner.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/src/lerobot/rl/learner.py): every transition to the online buffer, intervention transitions also to the offline buffer; learning starts after `online_step_before_learning`.
- [rl/data_sources/data_mixer.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/src/lerobot/rl/data_sources/data_mixer.py) and [rl/train_rl.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/src/lerobot/rl/train_rl.py): 50/50 online/offline batches.
- [sac/configuration_sac.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/src/lerobot/rl/algorithms/sac/configuration_sac.py), [gaussian_actor/configuration_gaussian_actor.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/src/lerobot/policies/gaussian_actor/configuration_gaussian_actor.py): SAC and buffer defaults.
- [examples/tutorial/rl/hilserl_example.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/examples/tutorial/rl/hilserl_example.py): a minimal, readable actor/learner script, good for students who want to see the loop in plain Python.
- [rl/gym_manipulator.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/src/lerobot/rl/gym_manipulator.py), [processor/hil_processor.py](https://github.com/huggingface/lerobot/blob/45cca1a6ba368cf29cc8b4e0f49a1db875d15f61/src/lerobot/processor/hil_processor.py): reward 0 from the hardware, success, classifier and gripper-penalty steps.
- Luo et al., *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*, arXiv:2410.21845 (HIL-SERL).

#### Intentionally left out

- **The full downloaded config files.** The example `env_config.json` / `train_config.json` are on the Hugging Face Hub and could not be fetched from here; the blocks above show only the keys for this task. Merge them into the downloaded files.
- **SAC mathematics** beyond the two target equations (entropy tuning, tanh-squashed Gaussian, UTD ratio).
- **Leader arm as the intervention device.** Appendix D and the guide mention it, but in the code checked only the gamepad and keyboard teleops report success / intervention events.
- **TOPReward, SARM, Robometer.** Appendix D lists TOPReward as a reward source. These models exist in LeRobot, but nothing in `lerobot/rl/` calls them, so they do not feed the real-arm training loop today.
- **Simulated HIL-SERL** (gym-hil, Franka in MuJoCo): useful practice before touching hardware, but not the SO101.