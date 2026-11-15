# SO-101 Lab: Training an ACT Policy to Lift a Cube (LeRobot)

Nov 2, 2026 · @weecheehong

## Lab overview

In this lab, learners teach a real SO-101 follower arm to pick up a cube and lift it, by showing it how with a leader arm and then training an ACT (Action Chunking with Transformers) policy on those demonstrations using Hugging Face LeRobot. By the end, the arm runs the task on its own from camera images and joint readings, with no human holding the leader arm.

&#91;embedded content: Lab at a glance · 6 stages, 1 improvement loop\]

Data quality is the main lever: when the policy fails, the fix is usually better or more demonstrations at the positions where it fails, not a different model.

**Learning outcomes.** On completion, learners can:

1. Set up, wire and calibrate an SO-101 leader/follower pair and its cameras for data collection.
2. Explain how imitation learning and ACT turn demonstrations into a control policy.
3. Record a clean, consistent teleoperated dataset in LeRobot format and inspect it.
4. Train an ACT policy with the LeRobot training script and read its training loss.
5. Run the trained policy on the real arm, score its success rate and name its failure modes.

**Who it is for.** Beginner adult learners who can use a terminal and run Python scripts. No prior robotics or reinforcement learning is assumed; basic neural-network vocabulary (encoder, transformer, loss) helps.

**Group size.** One leader/follower pair per team of 3 to 4 learners. Rotate roles every 10 episodes: teleoperator, recorder (keyboard), scene resetter, logbook keeper.

**Time.** About 12 contact hours, split into four sessions of 3 hours. Each session produces an output the next one needs, so a team can stop at any session boundary.

| Session | Steps | Output handed to the next session |
| --- | --- | --- |
| A. Build and calibrate (3 h) | Hardware setup, software setup, Steps 1 to 2 | Calibrated arms; known USB ports; calibration files |
| B. Teleoperate and record (3 h) | Steps 3 to 5 | Dataset of 50 cube-lift episodes, checked and pushed to the Hub |
| C. Train (3 h class + unattended run) | Step 6 | ACT checkpoint (`pretrained_model` folder) |
| D. Evaluate and analyse (3 h) | Step 7, troubleshooting, write-up | Evaluation dataset, success-rate table, failure-mode notes |

**Before you start.** Instructors should complete the whole guide once on their own rig before the first class: the motor ID step and calibration are where most first-time teams lose time.

## Concept overview: imitation learning and ACT

ACT learns to map what the robot sees and feels right now (camera images plus six joint angles) to the next 100 joint targets, by copying human demonstrations. Predicting a whole chunk of motion at once, instead of one step at a time, is what lets it learn a cube lift from about 50 demonstrations.

### Imitation learning in plain terms

Imitation learning trains a policy from examples of an expert doing the task. Its simplest form, behaviour cloning, is ordinary supervised learning: the input is the observation at time t and the label is the action the human took at time t.

In this lab, each recorded frame (30 per second) holds:

| Field in the dataset | What it is | Shape |
| --- | --- | --- |
| `observation.state` | Follower joint positions: 5 joints in degrees, gripper 0 to 100 | 6 |
| `observation.images.front` | Fixed front/top camera frame | 3 × 480 × 640 |
| `observation.images.wrist` | Wrist camera frame | 3 × 480 × 640 |
| `action` | Leader arm joint positions at that instant: the target sent to the follower | 6 |

|  | Reinforcement learning | Imitation learning (this lab) |
| --- | --- | --- |
| Learning signal | A reward you design | Human demonstrations |
| Where data comes from | The robot's own trial and error, usually in simulation | Teleoperation with the leader arm |
| Typical data size | Millions of simulated steps | 50 episodes, about 15 to 20 minutes of real data |
| Main risk | Reward hacking, sim-to-real gap | Poor or inconsistent demonstrations |

### Why plain behaviour cloning struggles

1. **Compounding errors.** A one-step policy makes about 600 decisions in a 20-second episode. Each small error moves the arm to a state the demonstrations never showed, so the next error is larger.
2. **Inconsistent humans.** Demonstrators pause, hesitate and take different paths. A model trained to predict one averaged action can end up between two good paths, which is a bad path.

### ACT's three ideas

1. **Action chunking.** The policy predicts the next k = 100 actions in one pass (3.3 seconds at 30 Hz). The number of decisions per episode drops 100-fold, and each chunk is a smooth, committed motion.
2. **A style variable (CVAE).** During training, an extra encoder sees the real future actions and compresses their "style" (fast or slow, this path or that) into a 32-number latent z. The decoder learns to use z, so demonstrator variation is explained by z instead of blurring the actions. At inference there is no future to look at, so z is set to zero, the average style.
3. **Temporal ensembling (optional).** Query the policy every step and blend the overlapping chunks, weighting older predictions more. LeRobot leaves this off by default and instead plays the whole chunk, then queries again.

The training loss is an L1 reconstruction term on the action chunk plus a KL term that keeps z close to a standard normal:

```latex
\mathcal{L} = \frac{1}{k}\sum_{i=1}^{k} \lVert \hat{a}_{t+i} - a_{t+i} \rVert_1 \; + \; \beta \, D_{KL}\big(q(z \mid a_{t:t+k}, s_t) \,\Vert\, \mathcal{N}(0, I)\big), \quad \beta = 10
```

With temporal ensembling switched on, the action executed at step t is a weighted average over every chunk that covers step t, where i = 0 is the oldest prediction:

```latex
a_t = \frac{\sum_i w_i \, A_t^{(i)}}{\sum_i w_i}, \qquad w_i = e^{-m \cdot i}
```

**Worked example.** One episode lasts 20 s at 30 Hz, so 600 control steps. With LeRobot's defaults (`chunk_size=100`, `n_action_steps=100`) the policy runs 600 / 100 = 6 times per episode. With temporal ensembling (`temporal_ensemble_coeff=0.01`, which forces `n_action_steps=1`) it runs 600 times, and each executed action blends up to 100 predictions. The second is smoother but needs 100 times more inference per second.

&#91;embedded content: How ACT executes chunks · default vs temporal ensembling\]

### ACT architecture as configured in LeRobot

&#91;embedded content: ACT data flow · training vs inference\]

With two 480 × 640 cameras, ResNet-18 turns each image into a 15 × 20 feature map, so 300 tokens per camera. Add one joint-state token and one latent token, and the encoder reads 602 tokens. The decoder answers 100 learned queries with 100 actions of 6 numbers each.

| ACT setting (`--policy.*`) | LeRobot default | What it controls |
| --- | --- | --- |
| `chunk_size` | 100 | Actions predicted per pass (k) |
| `n_action_steps` | 100 | Actions executed before the next query |
| `temporal_ensemble_coeff` | None (off) | m in the ensembling weights; needs `n_action_steps=1` |
| `vision_backbone` | resnet18, ImageNet weights | Image encoder |
| `dim_model` / `n_heads` | 512 / 8 | Transformer width and attention heads |
| `n_encoder_layers` / `n_decoder_layers` | 4 / 1 | Depth (1 decoder layer matches the original code's behaviour) |
| `use_vae` / `latent_dim` | true / 32 | CVAE style variable |
| `kl_weight` | 10 | β in the loss |
| `dropout` | 0.1 | Transformer dropout |
| `optimizer_lr` / `optimizer_lr_backbone` | 1e-5 / 1e-5 | AdamW learning rates |

ACT is the first policy LeRobot recommends trying: about 50 demonstrations often reach a high success rate, and training takes a few hours on one GPU. The original paper reports 80 to 90% success on six fine bimanual tasks from about 10 minutes of demonstrations each ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)).

## Hardware setup

One lab station is a leader arm, a follower arm, two USB cameras and a PC with an NVIDIA GPU, laid out so that nothing in the scene moves between recording and evaluation. A policy trained on one camera angle will fail if the camera is bumped, so fix and mark everything before the first recording.

### Bill of materials (per station)

| Item | Qty | Specification and notes |
| --- | --- | --- |
| SO-101 follower arm, assembled | 1 | 6 × Feetech STS3215, 7.4 V, 1/345 gearing (C001). A 12 V "Pro" servo variant also exists. |
| SO-101 leader arm, assembled | 1 | Always 7.4 V. 1 × C001 (1/345, shoulder lift), 2 × C044 (1/191, base and elbow), 3 × C046 (1/147, wrist flex, wrist roll, gripper) |
| Motor control board (bus servo adapter) | 2 | One per arm. On a Waveshare board, set both jumpers to channel B (USB). |
| DC power supply | 2 | 5 V supply for 7.4 V servos. 12 V, 5 A or more only for a 12 V follower. Never cross them. |
| USB-C data cable | 2 | Must carry data, not charge only |
| Table clamp | 4 | Two per arm base |
| Wrist camera | 1 | 32 × 32 mm UVC module in the SO-101 wrist mount |
| Front (scene) camera | 1 | USB webcam that does 640 × 480 at 30 fps, on a clamp or small tripod |
| PC | 1 | Ubuntu 22.04 or 24.04 recommended; NVIDIA GPU with 8 GB+ VRAM for local training; 3 free USB ports on separate controllers if possible |
| Cube | 2 to 3 | 2.5 to 3 cm, matte, a colour that contrasts with the table (for example red on a white mat) |
| Table mat and masking tape | 1 each | Plain, non-reflective surface; tape marks for cube positions and camera feet |
| Phillips screwdrivers | 1 set | Sizes #0 and #1 |

### Wiring each arm

1. Do **Step 1 (motor IDs)** before you daisy-chain the motors. New STS3215 motors all ship with ID 1, so a full chain cannot be addressed until each motor has its own ID.
2. Chain the 3-pin cables in order: control board → shoulder pan (ID 1) → shoulder lift (2) → elbow flex (3) → wrist flex (4) → wrist roll (5) → gripper (6).
3. Plug the DC supply into the control board's barrel jack, then the USB-C cable into the PC.
4. Label every board, supply and USB cable **LEADER** or **FOLLOWER** with tape. Most first-session faults are a swapped cable or a supply on the wrong arm.
5. Route cables so they cannot snag when the arm moves through its full range, especially around the wrist roll joint.

&#91;embedded content: Wiring for one arm · repeat for the other\]

### Workspace layout

&#91;embedded content: Station layout · top view\]

- **Follower.** Clamp it at the table edge, facing into the workspace. Mark its base outline with tape.
- **Leader.** Clamp it beside the follower, about 40 cm away, so the teleoperator stands outside the front camera's view.
- **Front camera.** Mount it about 50 to 60 cm in front of the follower, looking down at roughly 30 to 45 degrees, so the gripper, the cube and the lift height are all in frame. Tape its feet.
- **Wrist camera.** Fit it in the wrist mount so the fixed claw and the cube are both visible when the gripper is open.
- **Cube zone.** Mark 5 cube positions with tape about 15 to 25 cm in front of the follower base: one centre spot and four spots about 5 cm from it. You will record 10 demonstrations at each.
- **Lighting.** Use steady overhead room light. Avoid windows behind the scene and strong shadows from people standing nearby; keep the same lighting for evaluation.

### Safety

- The follower is strong enough to pinch fingers. Keep hands out of its workspace whenever it is under policy control.
- The fastest stop is to switch off or unplug the **follower power supply**. Ctrl+C in the terminal also stops the script, and LeRobot disables torque on disconnect by default.
- If a motor is stalled (for example, the gripper squeezing hard against the table), stop and release it. A stalled STS3215 heats up quickly.
- Tie back long hair and remove lanyards before teleoperating.

### Pre-flight checklist

- [ ] Both arms clamped; base outlines taped
- [ ] Correct supply on each arm (check the label voltage)
- [ ] Leader and follower cables labelled
- [ ] Front camera taped in place; wrist camera seated
- [ ] Five cube positions marked
- [ ] Pinch-point and stop briefing given to the whole team

## Software setup

Install LeRobot v0.6.1 (the latest tagged release when this guide was written) into a Python 3.12 conda environment with the Feetech, recording and training extras. Every later command reads its ports, IDs and camera settings from one station file, so learners edit values in one place.

| Component | This guide uses | Notes |
| --- | --- | --- |
| OS | Ubuntu 22.04 or 24.04 | Windows 11 works; see the Windows notes below |
| Python | 3.12 | LeRobot 0.6 requires 3.12 or newer |
| LeRobot | v0.6.1, installed from source | Pinned so commands match this guide |
| PyTorch | Pulled in by LeRobot (2.7 up to 2.11) | Must be a CUDA build for GPU training |
| NVIDIA driver | 570 or newer | Needed for CUDA 12.8 wheels and RTX 50-series cards |

### Install on Ubuntu

1. Check the GPU driver. The table should list your GPU and a driver version of 570 or newer.

   ```bash
   nvidia-smi
   ```
2. Install Miniforge (conda), then close and reopen the terminal.

   ```bash
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
3. Create and activate the environment. Run the `activate` line in every new terminal.

   ```bash
   conda create -y -n lerobot python=3.12
   conda activate lerobot
   ```
4. Install ffmpeg for video encoding and decoding.

   ```bash
   conda install -y ffmpeg -c conda-forge
   # If `ffmpeg -encoders | grep svtav1` prints nothing, or torchcodec complains, pin it:
   # conda install -y ffmpeg=7.1.1 -c conda-forge
   ```
5. Get LeRobot, pin the release, and install it with the extras this lab needs.

   ```bash
   cd ~
   git clone https://github.com/huggingface/lerobot.git
   cd lerobot
   git checkout v0.6.1
   pip install -e ".[core_scripts,training,feetech]"
   ```
6. Let your user open the arms' serial ports without `sudo`, then log out and back in.

   ```bash
   sudo usermod -aG dialout $USER
   # Quick fix for this session only (needed again after every replug):
   # sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
   ```
7. Verify the install. Expect `0.6.1`, a torch version, and `True` for CUDA.

   ```bash
   python -c "import lerobot, torch; print('lerobot', lerobot.__version__); print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
   lerobot-info
   ```
8. Log in to the Hugging Face Hub. Create a token with **write** access at huggingface.co → Settings → Access Tokens, then:

   ```bash
   hf auth login
   HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
   echo $HF_USER
   ```
9. Optional: log in to Weights & Biases to watch training curves in a browser.

   ```bash
   wandb login
   ```

### Create the station file

Save this as `~/so101_lab/station.env`. Leave the port and camera values as they are for now; you will fill them in during Steps 1 and 3.

```bash
# ~/so101_lab/station.env  --  SO-101 ACT lab, one file per station
export STATION=st03                          # your station number
export FOLLOWER_PORT=/dev/ttyACM0            # from Step 1 (lerobot-find-port)
export LEADER_PORT=/dev/ttyACM1              # from Step 1
export FOLLOWER_ID=${STATION}_follower       # names the calibration file
export LEADER_ID=${STATION}_leader
export FRONT_CAM=0                           # from Step 3 (lerobot-find-cameras opencv)
export WRIST_CAM=2                           # from Step 3
export CAMERAS="{ front: {type: opencv, index_or_path: ${FRONT_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30} }"
export TASK="Pick up the red cube and lift it"
export HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
export DATASET=${HF_USER}/so101_cube_lift_${STATION}
export POLICY=${HF_USER}/act_so101_cube_lift_${STATION}
```

Load it at the start of every terminal session:

```bash
conda activate lerobot
source ~/so101_lab/station.env
```

### Windows notes

- Use the **Miniforge Prompt** for steps 2 to 5 and 7 to 9; the commands are the same. Skip step 6.
- Serial ports appear as `COM3`, `COM4` and so on; use those wherever this guide shows `/dev/ttyACM0`.
- If CUDA prints `False`, or you have an RTX 50-series card, reinstall a CUDA 12.8 PyTorch inside LeRobot's supported range:

  ```powershell
  pip install --force-reinstall "torch<2.12" "torchvision<0.27" --index-url https://download.pytorch.org/whl/cu128
  ```
- In PowerShell, continue long commands with a backtick (`` ` ``) instead of a backslash, and use this as the station file (`station.ps1`, loaded with `. .\station.ps1`):

  ```powershell
  $env:STATION = "st03"
  $env:FOLLOWER_PORT = "COM3"
  $env:LEADER_PORT = "COM4"
  $env:FOLLOWER_ID = "$($env:STATION)_follower"
  $env:LEADER_ID = "$($env:STATION)_leader"
  $env:CAMERAS = "{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }"
  $env:TASK = "Pick up the red cube and lift it"
  $env:HF_USER = "your-hf-username"
  $env:DATASET = "$($env:HF_USER)/so101_cube_lift_$($env:STATION)"
  $env:POLICY = "$($env:HF_USER)/act_so101_cube_lift_$($env:STATION)"
  ```

  Then refer to them as `$env:FOLLOWER_PORT` and so on wherever the bash commands below use `$FOLLOWER_PORT`.

## Step 1: Find the USB ports and set the motor IDs

This step tells the PC which serial port is which arm, and gives each of the 12 motors its own ID (1 to 6 per arm) at the bus speed LeRobot expects (1,000,000 baud). Motor IDs are stored in the motor's memory, so this is done once per arm, not every lesson.

### 1a. Find each arm's port

Plug in both control boards (USB and power), then run:

```bash
lerobot-find-port
```

When prompted, unplug the **follower** USB cable and press Enter. The script reports the port that disappeared. Plug it back in, run the command again, and unplug the **leader** this time.

**Worked example (Ubuntu):**

```text
Finding all available ports for the MotorsBus.
Ports before disconnecting: ['/dev/ttyACM0', '/dev/ttyACM1']
Remove the USB cable from your MotorsBus and press Enter when done.

The port of this MotorsBus is '/dev/ttyACM0'
Reconnect the USB cable.
```

Here the follower is `/dev/ttyACM0`. Write both ports into `station.env` (`FOLLOWER_PORT`, `LEADER_PORT`) and reload it with `source ~/so101_lab/station.env`.

> Ports can swap after a reboot or when cables are moved to different sockets. Always plug each arm into the same labelled USB socket, and re-run `lerobot-find-port` if a later command cannot find the motors.

### 1b. Set the motor IDs (follower)

Skip 1b and 1c if your kit arrived with the IDs already set; Step 2 will confirm it by finding all six motors.

1. Disconnect every 3-pin cable from the control board, so no motor is on the bus.
2. Run:

   ```bash
   lerobot-setup-motors \
       --robot.type=so101_follower \
       --robot.port=$FOLLOWER_PORT
   ```
3. The script asks for the motors one at a time, starting from the gripper:

   ```text
   Connect the controller board to the 'gripper' motor only and press enter.
   'gripper' motor id set to 6
   Connect the controller board to the 'wrist_roll' motor only and press enter.
   ```
4. For each prompt, connect **only that motor** directly to the board with a 3-pin cable, press Enter, and wait for the "id set to" message. Then unplug the board end of that cable and move on. The order is gripper (6), wrist roll (5), wrist flex (4), elbow flex (3), shoulder lift (2), shoulder pan (1).
5. When the script ends, daisy-chain the motors as in the wiring diagram, with the board connected to the shoulder pan motor (ID 1).

### 1c. Set the motor IDs (leader)

Repeat 1b on the leader arm:

```bash
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT
```

**Check before moving on:** if the script stops with a connection error, check the three usual suspects: the power supply plug, the USB cable, and the 3-pin cable at both ends. On a Waveshare board, confirm both jumpers are on channel B.

## Step 2: Calibrate the follower and the leader

Calibration makes the leader and follower report the same numbers when they are in the same physical pose. Without it, the follower will not copy the leader faithfully, and a policy trained on one arm will not run on another. Each arm is calibrated once and saved under its `id`.

### Pose to start from

&#91;embedded content: Mid-range calibration pose · side view\]

### 2a. Calibrate the follower

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID
```

1. The script prints a pose diagram and waits with the message `Move the joints to the middle of their range of motion and press ENTER....` Torque is off, so the arm moves freely by hand.
2. Put the arm in the mid-range pose shown above and press **Enter**. This records the zero (homing) offset for each joint.
3. The script prints a live table with `NAME | MIN | POS | MAX` for each joint. Move **each joint, one at a time, slowly through its full range** until it reaches its mechanical stop in both directions: shoulder pan, shoulder lift, elbow flex, wrist flex, gripper (fully closed and fully open). Wrist roll is a full-turn joint and is not recorded, so you can leave it.
4. Watch the MIN and MAX columns grow apart as each joint moves. When every joint has been moved end to end, press **Enter**.
5. The script prints `Calibration saved to ...`. The file is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<FOLLOWER_ID>.json`.

### 2b. Calibrate the leader

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID
```

Follow the same five steps, starting from the **same** mid-range pose. The leader's file is saved under `.../calibration/teleoperators/so_leader/<LEADER_ID>.json`.

### Rules that save time later

- Use the same `--robot.id` and `--teleop.id` in every later command. A different id means LeRobot looks for a different calibration file and asks you to calibrate again.
- If you are asked `Press ENTER to use provided calibration file ... or type 'c'`, press **Enter** to keep the existing calibration. Type `c` only if the arm was reassembled or a motor was replaced.
- Recalibrating either arm after you record data changes what the numbers mean. If that happens, record a new dataset.
- Back up both JSON files to the team's shared folder at the end of Session A.

## Step 3: Teleoperate and check the cameras

Before recording anything, prove three things: the follower copies the leader smoothly, both cameras stream at 30 fps, and a person watching only the camera images could do the cube lift. Recording with any of these broken wastes the whole session.

### 3a. Teleoperate without cameras

Hold the leader roughly in the follower's current pose before you start, because the follower moves straight to the leader's position when the script connects.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID
```

Move the leader slowly at first, then at normal speed. Every team member should do 3 to 5 practice lifts from the centre spot. Press **Ctrl+C** to stop; torque is released on exit, so support the follower as it relaxes.

What good looks like: the follower tracks the leader with no visible lag, the gripper closes fully on the cube, and nothing jerks at the ends of each joint's range. If one joint moves the wrong way or jumps, recalibrate that arm (Step 2).

### 3b. Identify the two cameras

```bash
lerobot-find-cameras opencv
```

The script lists every camera it finds (`Camera #0`, `Camera #1` ... with an id or path, resolution and fps) and saves one image from each to `outputs/captured_images`. Open the images to see which index is the front camera and which is the wrist camera. Write the two values into `station.env` as `FRONT_CAM` and `WRIST_CAM`, then reload it.

On Linux one USB camera often appears as two `/dev/video` nodes, so indices such as 0 and 2 are normal. You can also use a stable path, for example `/dev/v4l/by-id/usb-...-video-index0`, as `index_or_path`; it does not change between reboots.

### 3c. Teleoperate with cameras and the live viewer

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true
```

`--display_data=true` opens a Rerun window with both camera streams and a plot of every joint. Check:

- [ ] Both camera views update smoothly, without freezing or tearing
- [ ] The cube is visible in the front view at all 5 taped spots, and in the wrist view as the gripper approaches
- [ ] The lift height is visible in the front view (the cube clearly leaves the table)
- [ ] Nobody's body or the leader arm appears in the front view
- [ ] Joint plots move smoothly with no spikes

If one camera fails or the frame rate drops with both connected, the USB bus is probably saturated. Move the cameras to different USB controllers, or ask the camera for compressed frames by adding `fourcc: MJPG` to each camera's settings in `CAMERAS`, for example `{type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}`.

## Step 4: Record the cube-lift demonstrations

Record 50 episodes, 10 at each of the 5 taped spots, with every episode following the same script. ACT copies what it sees, so consistent demonstrations matter more than fast or clever ones.

### The episode script

&#91;embedded content: One demonstration · 5 recorded phases, then an unrecorded reset\]

1. **Start pose.** Gripper open, arm raised about 10 cm above the table and just behind the cube zone. Take a photo of it and match it every time.
2. **Reach.** Move straight to the cube in one smooth motion; no pausing, no hunting around.
3. **Grasp.** Come down from above, centre the claws on the cube, close fully.
4. **Lift.** Raise the cube about 10 cm, straight up.
5. **Hold.** Keep it still for about a second, then the recorder presses **→** to end the episode.
6. **Reset (not recorded).** Lower the cube, open the gripper, return to the start pose. The scene resetter moves the cube to the next spot and takes their hands out of view before the next episode starts.

**Spot order.** Cycle through the spots (centre, left, right, near, far, then repeat) rather than doing 10 in a row at one spot, so tiredness and practice are spread evenly across positions.

**If a grasp fails,** press **←** to throw the episode away and record it again. Do not correct mid-episode; at this stage you want clean successes only.

### 4a. Test run (2 episodes, kept local)

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true \
    --dataset.repo_id=${DATASET}_test \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes=2 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false
```

Check that the script finishes, encodes the videos and prints no errors, and that the dataset folder exists at `~/.cache/huggingface/lerobot/<HF_USER>/so101_cube_lift_<STATION>_test`. Then delete the test folder.

### 4b. Record the real dataset (50 episodes)

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true \
    --dataset.repo_id=$DATASET \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.private=true \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
```

| Flag | Value here | Why |
| --- | --- | --- |
| `--dataset.episode_time_s` | 20 | Upper limit; most lifts finish in under 15 s and are ended early with → |
| `--dataset.reset_time_s` | 10 | Time to lower the cube, move it to the next spot and return to the start pose |
| `--dataset.num_episodes` | 50 | 10 per spot × 5 spots |
| `--dataset.single_task` | `$TASK` | The task sentence stored with every frame |
| `--dataset.private` | true | Keeps the uploaded dataset visible only to your account |
| `--dataset.streaming_encoding` | true | Encodes video while recording, so the end of session is not a long wait |
| `--dataset.fps` | 30 (default) | Must match the cameras' fps |

### Keyboard controls while recording

| Key | Action |
| --- | --- |
| → or `n` | End the current episode (or the reset period) early and move on |
| ← or `r` | Discard the current episode and record it again |
| Esc or `q` | Stop the session, finish encoding, and upload to the Hub |

On Ubuntu with Wayland (the default desktop), LeRobot reads these keys from the terminal, so keep the terminal window focused when you press them. On Windows and on X11 they work from any window.

### If the session is interrupted

Resume into the same dataset, giving the number of **additional** episodes and the local dataset folder:

```bash
lerobot-record \
    ...same robot, camera and teleop flags as 4b... \
    --dataset.repo_id=$DATASET \
    --dataset.single_task="$TASK" \
    --dataset.root=$HOME/.cache/huggingface/lerobot/$DATASET \
    --dataset.num_episodes=20 \
    --resume=true
```

### Team logbook (one row per episode)

| Episode | Spot | Teleoperator | Kept or re-recorded | Notes |
| --- | --- | --- | --- | --- |
| 0 | Centre |  |  |  |
| 1 | Left |  |  |  |
| 2 | Right |  |  |  |

One teleoperator for all 50 gives the most consistent data. If your team rotates for practice, rotate after each full round of 5 spots and note who recorded what; it helps when you analyse failures in Step 7.

## Step 5: Inspect the dataset before training

Spend 20 minutes checking the data before committing hours of GPU time. A training run cannot fix a dataset with a hidden camera, wrong task text or a handful of bad episodes, but deleting those episodes now takes one command.

### 5a. Know what was saved

The dataset is in LeRobot format v3.0, both on the Hub (`$DATASET`) and locally at `~/.cache/huggingface/lerobot/$DATASET`:

```text
so101_cube_lift_st03/
├── meta/
│   ├── info.json          # fps, features and shapes, episode and frame counts
│   ├── stats.json         # min, max, mean, std per feature (used to normalise for training)
│   ├── tasks.parquet      # the task sentence(s)
│   └── episodes/chunk-000/file-000.parquet   # one row per episode: length, frame range, video offsets
├── data/chunk-000/file-000.parquet           # one row per frame: observation.state, action, timestamps
└── videos/
    ├── observation.images.front/chunk-000/file-000.mp4
    └── observation.images.wrist/chunk-000/file-000.mp4
```

### 5b. Run the quality-check script

Save the script from the appendix as `~/so101_lab/check_dataset.py`, then run:

```bash
python ~/so101_lab/check_dataset.py --repo-id $DATASET
```

Read the output against this checklist:

- [ ] 50 episodes, 2 cameras (`observation.images.front`, `observation.images.wrist`), state and action of shape 6
- [ ] Total recording time of roughly 8 to 15 minutes
- [ ] No episode flagged as shorter than 4 s (a slip of the → key) or longer than 19.5 s (the 20 s limit was hit, so the lift was probably not finished)
- [ ] Each joint's action range looks plausible; a range near zero means that joint was barely used, which is fine for wrist roll but not for the gripper

### 5c. Watch episodes

Locally, with the Rerun viewer (cameras plus every joint plotted over time):

```bash
lerobot-dataset-viz --repo-id $DATASET --episode-index 0
```

Or in a browser: open the [LeRobot dataset visualiser](https://huggingface.co/spaces/lerobot/visualize_dataset) and enter your dataset id. A private dataset may not load there; use the local viewer instead.

Watch at least one episode per spot. Look for hands in the frame during an episode, a cube outside the front view, motion blur, and long pauses.

### 5d. Replay a demonstration on the follower

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --dataset.repo_id=$DATASET \
    --dataset.episode=0
```

Put the cube on episode 0's spot first. The follower repeats the recorded actions exactly, with its eyes closed. **Discussion point:** move the cube 3 cm and replay again. The replay misses, because it ignores the cameras; the ACT policy you train next looks at the images every chunk, which is why it can adapt.

### 5e. Remove bad episodes

If you found bad episodes, delete them and save a cleaned copy. Note that indices are renumbered afterwards.

```bash
lerobot-edit-dataset \
    --repo_id $DATASET \
    --new_repo_id ${DATASET}_clean \
    --operation.type delete_episodes \
    --operation.episode_indices "[7, 23]" \
    --push_to_hub true
```

Then set `DATASET=${HF_USER}/so101_cube_lift_${STATION}_clean` in `station.env` and reload it, so training uses the cleaned copy.

## Step 6: Train the ACT policy

Train with LeRobot's default ACT settings: 100,000 steps at batch size 8, which LeRobot describes as a few hours on a single GPU. Start the run at the end of Session C and let it finish unattended; Session D begins with a trained checkpoint.

### Choose where to train

| Option | When to use it | What you need |
| --- | --- | --- |
| Lab PC (local GPU) | Default. Dataset is already on disk | NVIDIA GPU, 8 GB+ VRAM, CUDA PyTorch (Software setup, step 7) |
| Google Colab | No suitable local GPU | Google account, GPU runtime, dataset pushed to the Hub |
| Hugging Face Jobs | Paid cloud GPU from the same command | HF account with billing; add `--job.target=a10g-small` |

### 6a. Start training on the lab PC

Run training inside `tmux` so it keeps going if the terminal or SSH session closes:

```bash
tmux new -s train
conda activate lerobot
source ~/so101_lab/station.env

lerobot-train \
    --dataset.repo_id=$DATASET \
    --policy.type=act \
    --policy.device=cuda \
    --policy.repo_id=$POLICY \
    --policy.private=true \
    --output_dir=outputs/train/act_so101_cube_lift_${STATION} \
    --job_name=act_so101_cube_lift_${STATION} \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=20000 \
    --wandb.enable=false
```

Detach with **Ctrl+B** then **D**; reattach later with `tmux attach -t train`. Set `--wandb.enable=true` if you ran `wandb login` and want curves in the browser.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--policy.type` | act | Use the ACT policy with its defaults (see the concept table) |
| `--policy.repo_id` | `$POLICY` | Where the final model is pushed on the Hub |
| `--batch_size` | 8 | Training samples per step; halve it if you get a CUDA out-of-memory error |
| `--steps` | 100000 | Total optimiser updates |
| `--save_freq` | 20000 | A checkpoint every 20k steps, plus one at the end |
| `--output_dir` | `outputs/train/...` | Must not already exist, unless you are resuming |

**Worked example: how many times does ACT see each frame?** Suppose your 50 episodes average 12 s. That is 50 × 12 × 30 = 18,000 frames. Training draws 100,000 × 8 = 800,000 samples, so each frame is used about 800,000 / 18,000 ≈ 44 times (44 epochs). The `epch:` field in the log shows this number growing.

### 6b. Read the training log

Every 200 steps LeRobot prints one line like `step:2K smpl:16K ep:44 epch:0.89 loss:... grdn:... lr:1.0e-05 updt_s:... data_s:...`.

- `loss` is the L1 action error plus 10 × KL. Expect it to fall steeply over the first few thousand steps and then flatten. A loss that is flat or rising from the start points to a data or configuration problem.
- `lr` stays at 1.0e-05; ACT's default uses no learning-rate schedule.
- If `data_s` is much larger than `updt_s`, the GPU is waiting for video decoding; close other programs, or keep `num_workers` at its default of 4.
- A low training loss alone does not prove the policy works. Only the real-robot test in Step 7 does.

### 6c. Find the checkpoints

```text
outputs/train/act_so101_cube_lift_st03/checkpoints/
├── 020000/pretrained_model/   # config.json, model.safetensors, train_config.json, ...
├── 040000/ ...
├── 100000/pretrained_model/
└── last -> 100000            # always points to the newest checkpoint
```

The final model is also pushed to `$POLICY` on the Hub. To push an intermediate checkpoint for comparison in Step 7:

```bash
hf upload ${POLICY}_040k \
    outputs/train/act_so101_cube_lift_${STATION}/checkpoints/040000/pretrained_model
```

### 6d. Resume an interrupted run

```bash
lerobot-train \
    --config_path=outputs/train/act_so101_cube_lift_${STATION}/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

### 6e. Train on Google Colab instead

1. Make sure the dataset is on the Hub (it is, unless you recorded with `--dataset.push_to_hub=false`).
2. Open the official [ACT training notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb) and choose **Runtime → Change runtime type → GPU**.
3. Run its install cells, then log in with your Hub token (`hf auth login`).
4. Replace the training cell with the command from 6a, writing your real dataset and policy ids instead of `$DATASET` and `$POLICY`, and add `--save_checkpoint_to_hub=true`. Colab sessions can disconnect, and this pushes every checkpoint to the Hub as it is saved.
5. If the session drops, resume from the Hub in a new session: `lerobot-train --config_path=<your-hf-username>/<policy-name> --resume=true`.

For reference, the notebook's own guide quotes about 1.5 hours for 100k steps on an A100 at batch size 64.

## Step 7: Run the policy on the robot and score it

Run 25 scored trials, 5 at each taped spot, and report the success rate per spot. `lerobot-rollout` drives the follower from the trained policy; the leader is not used. Use the **same camera names, resolution and positions** as in recording, because the policy expects `observation.images.front` and `observation.images.wrist` exactly as it saw them in training.

### Before every rollout

- [ ] Cameras untouched since recording (check the tape marks); same lighting
- [ ] Follower placed by hand in the start pose before launching (it holds the pose it is in when the script connects)
- [ ] Cube on a taped spot; hands out of the workspace
- [ ] One person's hand near the follower power switch

### 7a. First look (30 seconds, nothing recorded)

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=$POLICY \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS" \
    --robot.max_relative_target=10 \
    --task="$TASK" \
    --duration=30 \
    --display_data=true
```

`--policy.path` takes either the Hub id or a local folder, for example `outputs/train/act_so101_cube_lift_${STATION}/checkpoints/last/pretrained_model`. `--robot.max_relative_target=10` caps how far any joint may move in one control step (10 degrees, or 10% of the gripper range), a sensible guard for a first run; remove it if motion looks sluggish.

### 7b. Scored evaluation (25 trials, recorded)

```bash
lerobot-rollout \
    --strategy.type=episodic \
    --policy.path=$POLICY \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS" \
    --dataset.repo_id=${HF_USER}/eval_so101_cube_lift_${STATION} \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=15 \
    --dataset.private=true \
    --display_data=true
```

Between trials the follower returns to the pose it started in, and the scene resetter moves the cube to the next spot. Use the same keys as in recording: → ends a trial early once the result is clear, ← discards and repeats a trial (for example if someone bumped the table), Esc stops. The recorded evaluation dataset lets you replay every failure afterwards with `lerobot-dataset-viz`.

### 7c. Score each trial

| Result | Definition |
| --- | --- |
| Success | Cube grasped and lifted clear of the table (about 5 cm or more) and still held when the trial ends |
| Partial | Cube grasped and lifted but dropped, or pushed out of the spot before a successful lift |
| Fail | No grasp: missed, hovered, froze, or hit the table |

**Results table (fill in)**

| Spot | Trials | Success | Partial | Fail | Success rate | Most common failure |
| --- | --- | --- | --- | --- | --- | --- |
| Centre | 5 |  |  |  |  |  |
| Left | 5 |  |  |  |  |  |
| Right | 5 |  |  |  |  |  |
| Near | 5 |  |  |  |  |  |
| Far | 5 |  |  |  |  |  |
| **All** | **25** |  |  |  |  |  |

**Worked example of the arithmetic.** If a team scores 4, 3, 5, 2 and 4 successes on the five spots, the overall rate is 18 / 25 = 72%, and the weakest spot (2 / 5 = 40%) is where to record 10 more demonstrations before retraining.

### 7d. Analyse and improve

Answer these in the team write-up:

1. Which spot is weakest, and what do its failures have in common in the evaluation videos?
2. Is the failure in **perception** (goes to the wrong place), **grasping** (right place, bad grip) or **timing** (hesitates, jerks at chunk boundaries)?
3. Does an earlier checkpoint do better? Run 7b with `--policy.path=outputs/train/act_so101_cube_lift_${STATION}/checkpoints/040000/pretrained_model` and a new dataset id (for example ending in \_040k, since each evaluation needs its own), and compare.
4. Does smoother execution help? Rerun 7b with a new dataset id (for example ending in \_ens) and temporal ensembling switched on at inference only:

   ```bash
   lerobot-rollout ...same flags as 7b... \
       --policy.temporal_ensemble_coeff=0.01 \
       --policy.n_action_steps=1
   ```
5. What one change to the data would you make next, and why? (Add the answer to the improvement loop in the overview diagram.)

## Troubleshooting

Most faults in this lab are cables, ports or a camera that moved. Check those three before changing any software setting.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `lerobot-find-port` shows no new port, or `Permission denied: /dev/ttyACM0` | Charge-only USB cable, or user not in the `dialout` group | Swap to a data cable; run `sudo usermod -aG dialout $USER` and log in again |
| `lerobot-setup-motors` cannot find the motor | No power to the board, more than one motor connected, Waveshare jumpers on A | Check the supply plug; connect exactly one motor; set both jumpers to B |
| Asked to calibrate again every time | A different `--robot.id` / `--teleop.id` from the one used in Step 2 | Use the ids from `station.env` everywhere |
| "Mismatch between calibration values in the motor and the calibration file" | Motors hold a different calibration from the file for this id | Press Enter to write the file's calibration to the motors; type `c` only if the arm was rebuilt |
| A follower joint moves the wrong way or sits offset from the leader | Arms calibrated from different poses | Recalibrate both arms from the same mid-range pose |
| Occasional "Incorrect status packet" errors, or a joint twitches | Loose 3-pin cable or a weak supply | Reseat every 3-pin cable along the chain; check the supply voltage label |
| Camera fails to open, or the wrong view appears after a reboot | Camera index changed | Rerun `lerobot-find-cameras opencv`, or use a `/dev/v4l/by-id/...` path |
| One camera freezes when both run, or recording warns fps is below 30 | USB bandwidth | Cameras on separate USB controllers; add `fourcc: MJPG`; try `--display_data=false` |
| →, ← or Esc do nothing during recording | Wayland reads keys from the terminal only | Click into the terminal window first, or use `n`, `r`, `q` |
| `lerobot-record` refuses a name starting with `eval_` | That prefix is reserved for policy evaluation | Record demonstrations under another name; evaluate with `lerobot-rollout` |
| `FileExistsError: Output directory ... already exists` | Training was started before with the same `--output_dir` | Use a new `--job_name` and `--output_dir`, or resume (Step 6d) |
| `CUDA out of memory` | Batch too large for the GPU | `--batch_size=4` (and expect a longer run) |
| `torch.cuda.is_available()` prints `False` | CPU-only PyTorch or an old driver | Update the driver to 570+; reinstall the cu128 PyTorch build (Windows notes) |
| ffmpeg or torchcodec errors when encoding or loading video | ffmpeg build mismatch | `conda install -y ffmpeg=7.1.1 -c conda-forge` |
| Upload to the Hub fails with 401 or 403 | Token without write access | Create a write token and run `hf auth login` again |
| Rollout fails with a missing image feature | Camera names differ from training | Use exactly the same `CAMERAS` string (`front`, `wrist`) as in recording |
| Policy hovers, freezes or reaches for the wrong place | Camera moved, lighting changed, or a start pose unlike the demos | Restore the scene exactly; start from the photographed start pose; add demos at weak spots |
| Motion jerks every 3 seconds or so | Switching from one 100-step chunk to the next | Try temporal ensembling at inference (Step 7d, question 4) |

## Appendix: reference code

Every script below targets LeRobot v0.6.1 and reads its settings from `station.env`, so the only file a team edits is that one. Keep them together in `~/so101_lab/`:

```text
~/so101_lab/
├── station.env              # ports, ids, cameras, names (Software setup)
├── so101_lab.sh             # A1: one command per lab step
├── check_dataset.py         # A2: dataset quality check (Step 5)
├── teleop_loop.py           # A3: teleoperation in plain Python
├── act_training_loop.py     # A4: ACT training loop in plain Python
└── act_inference_loop.py    # A5: ACT control loop in plain Python
```

A1 and A2 are for running the lab. A3 to A5 are for teaching: each shows, in a page of Python, what one LeRobot command does inside.

### A1. `so101_lab.sh`: one command per step

Make it executable once with `chmod +x ~/so101_lab/so101_lab.sh`, then run for example `~/so101_lab/so101_lab.sh record`. Extra flags are passed straight to LeRobot, so `./so101_lab.sh train --batch_size=4` works.

| Session | Commands |
| --- | --- |
| A | `ports`, `motors-follower`, `motors-leader`, `calib-follower`, `calib-leader` |
| B | `teleop`, `cameras`, `teleop-cam`, `record-test`, `record`, `resume-record <N>`, `check`, `viz [ep]`, `replay [ep]` |
| C | `train`, `resume-train` |
| D | `eval-quick`, `eval` |

```bash
#!/usr/bin/env bash
# so101_lab.sh -- one entry point for every step of the SO-101 ACT cube-lift lab (LeRobot v0.6.1)
# Usage:  ./so101_lab.sh <command> [extra lerobot flags]
# Reads ports, ids, cameras and names from $LAB_DIR/station.env (default ~/so101_lab/station.env).
set -euo pipefail

LAB_DIR="${LAB_DIR:-$HOME/so101_lab}"
# shellcheck source=/dev/null
source "$LAB_DIR/station.env"

ROBOT=(--robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID")
TELEOP=(--teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID")
CAMS=(--robot.cameras="$CAMERAS")
RUN_DIR="outputs/train/act_so101_cube_lift_${STATION}"
EVAL_DS="${EVAL_DS:-${HF_USER}/eval_so101_cube_lift_${STATION}}"
POLICY_PATH="${POLICY_PATH:-$POLICY}"

usage() {
  cat <<'USAGE'
Session A  ports | motors-follower | motors-leader | calib-follower | calib-leader
Session B  teleop | cameras | teleop-cam | record-test | record | resume-record <N> | check | viz [ep] | replay [ep]
Session C  train | resume-train
Session D  eval-quick | eval
Override for a checkpoint comparison:
  POLICY_PATH=outputs/train/.../checkpoints/040000/pretrained_model EVAL_DS=<user>/eval_..._040k ./so101_lab.sh eval
USAGE
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  ports)           lerobot-find-port ;;
  motors-follower) lerobot-setup-motors --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" "$@" ;;
  motors-leader)   lerobot-setup-motors --teleop.type=so101_leader --teleop.port="$LEADER_PORT" "$@" ;;
  calib-follower)  lerobot-calibrate "${ROBOT[@]}" "$@" ;;
  calib-leader)    lerobot-calibrate "${TELEOP[@]}" "$@" ;;
  teleop)          lerobot-teleoperate "${ROBOT[@]}" "${TELEOP[@]}" "$@" ;;
  cameras)         lerobot-find-cameras opencv ;;
  teleop-cam)      lerobot-teleoperate "${ROBOT[@]}" "${CAMS[@]}" "${TELEOP[@]}" --display_data=true "$@" ;;
  record-test)
    lerobot-record "${ROBOT[@]}" "${CAMS[@]}" "${TELEOP[@]}" --display_data=true \
      --dataset.repo_id="${DATASET}_test" --dataset.single_task="$TASK" \
      --dataset.num_episodes=2 --dataset.episode_time_s=20 --dataset.reset_time_s=10 \
      --dataset.push_to_hub=false "$@" ;;
  record)
    lerobot-record "${ROBOT[@]}" "${CAMS[@]}" "${TELEOP[@]}" --display_data=true \
      --dataset.repo_id="$DATASET" --dataset.single_task="$TASK" \
      --dataset.num_episodes=50 --dataset.episode_time_s=20 --dataset.reset_time_s=10 \
      --dataset.private=true --dataset.streaming_encoding=true --dataset.encoder_threads=2 "$@" ;;
  resume-record)
    n="${1:?give the number of ADDITIONAL episodes, e.g. ./so101_lab.sh resume-record 20}"; shift
    lerobot-record "${ROBOT[@]}" "${CAMS[@]}" "${TELEOP[@]}" --display_data=true \
      --dataset.repo_id="$DATASET" --dataset.single_task="$TASK" \
      --dataset.root="$HOME/.cache/huggingface/lerobot/$DATASET" \
      --dataset.num_episodes="$n" --dataset.episode_time_s=20 --dataset.reset_time_s=10 \
      --resume=true "$@" ;;
  check)           python "$LAB_DIR/check_dataset.py" --repo-id "$DATASET" "$@" ;;
  viz)             lerobot-dataset-viz --repo-id "$DATASET" --episode-index "${1:-0}" ;;
  replay)          lerobot-replay "${ROBOT[@]}" --dataset.repo_id="$DATASET" --dataset.episode="${1:-0}" ;;
  train)
    lerobot-train --dataset.repo_id="$DATASET" --policy.type=act --policy.device=cuda \
      --policy.repo_id="$POLICY" --policy.private=true \
      --output_dir="$RUN_DIR" --job_name="act_so101_cube_lift_${STATION}" \
      --batch_size=8 --steps=100000 --save_freq=20000 --wandb.enable=false "$@" ;;
  resume-train)
    lerobot-train --config_path="$RUN_DIR/checkpoints/last/pretrained_model/train_config.json" --resume=true "$@" ;;
  eval-quick)
    lerobot-rollout --strategy.type=base --policy.path="$POLICY_PATH" "${ROBOT[@]}" "${CAMS[@]}" \
      --robot.max_relative_target=10 --task="$TASK" --duration=30 --display_data=true "$@" ;;
  eval)
    lerobot-rollout --strategy.type=episodic --policy.path="$POLICY_PATH" "${ROBOT[@]}" "${CAMS[@]}" \
      --dataset.repo_id="$EVAL_DS" --dataset.single_task="$TASK" \
      --dataset.num_episodes=25 --dataset.episode_time_s=20 --dataset.reset_time_s=15 \
      --dataset.private=true --display_data=true "$@" ;;
  help|-h|--help)  usage ;;
  *)               echo "Unknown command: $cmd"; usage; exit 1 ;;
esac
```

### A2. `check_dataset.py`: dataset quality check

```python
"""check_dataset.py -- quick quality check of a recorded SO-101 LeRobot dataset (LeRobot v0.6.1).

Reads only the metadata (no video download), then prints counts, shapes, episode
lengths and per-joint action ranges, and flags episodes that look wrong.

Usage:  python check_dataset.py --repo-id <hf_user>/<dataset> [--root <local folder>]
"""
import argparse

import numpy as np
from lerobot.datasets import LeRobotDatasetMetadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="local dataset folder (default: the Hugging Face cache)")
    p.add_argument("--min-s", type=float, default=4.0, help="flag episodes shorter than this (seconds)")
    p.add_argument("--max-s", type=float, default=19.5, help="flag episodes longer than this (seconds)")
    args = p.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = meta.fps
    state = meta.features["observation.state"]
    action = meta.features["action"]

    print(f"Dataset    : {args.repo_id}")
    print(f"Episodes   : {meta.total_episodes}")
    print(f"Frames     : {meta.total_frames}  ({meta.total_frames / fps / 60:.1f} min at {fps} fps)")
    print(f"Cameras    : {meta.camera_keys}")
    print(f"State      : shape {state['shape']}  names {state.get('names')}")
    print(f"Action     : shape {action['shape']}")

    lengths_s = np.array(list(meta.episodes["length"]), dtype=float) / fps
    print(f"Episode s  : min {lengths_s.min():.1f}  mean {lengths_s.mean():.1f}  max {lengths_s.max():.1f}")
    flagged = [(i, s) for i, s in enumerate(lengths_s) if s < args.min_s or s > args.max_s]
    for i, s in flagged:
        print(f"  CHECK episode {i}: {s:.1f} s")
    if not flagged:
        print("  no episodes flagged by length")

    stats = meta.stats["action"]
    names = action.get("names") or [f"joint_{i}" for i in range(len(stats["min"]))]
    print("Action range per joint (min .. max, degrees; gripper 0-100):")
    for name, lo, hi in zip(names, np.ravel(stats["min"]), np.ravel(stats["max"])):
        print(f"  {name:<20} {lo:8.1f} .. {hi:8.1f}   span {hi - lo:6.1f}")


if __name__ == "__main__":
    main()
```

### A3. `teleop_loop.py`: teleoperation in plain Python

The whole of teleoperation is: read the leader's six joint positions, send them to the follower as goal positions, repeat 30 times a second. Run it with `python ~/so101_lab/teleop_loop.py` after loading `station.env`.

```python
"""teleop_loop.py -- what lerobot-teleoperate does, in 25 lines (LeRobot v0.6.1).

Reads the leader's joint positions and sends them to the follower as targets, 30 times a second.
Run after `source ~/so101_lab/station.env`. Stop with Ctrl+C.
"""
import os
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

follower = SO101Follower(SO101FollowerConfig(port=os.environ["FOLLOWER_PORT"], id=os.environ["FOLLOWER_ID"]))
leader = SO101Leader(SO101LeaderConfig(port=os.environ["LEADER_PORT"], id=os.environ["LEADER_ID"]))
follower.connect()
leader.connect()

try:
    while True:
        t0 = time.perf_counter()
        action = leader.get_action()   # {'shoulder_pan.pos': ..., ..., 'gripper.pos': ...}
        follower.send_action(action)   # the same dict becomes the follower's goal positions
        precise_sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
except KeyboardInterrupt:
    pass
finally:
    leader.disconnect()
    follower.disconnect()
```

### A4. `act_training_loop.py`: the ACT training loop

The key line for learners is the `delta_timestamps` request: each training sample asks the dataset for the next 100 actions, which is how action chunking enters training. Run 500 steps to watch the loss fall; use `lerobot-train` for the real run.

```python
"""act_training_loop.py -- what lerobot-train does for ACT, without the logging and checkpoint plumbing.

Adapted from LeRobot's examples/tutorial/act/act_training_example.py (LeRobot v0.6.1).
For teaching: run a few hundred steps and watch the loss fall. Use lerobot-train for the real run.
Run after `source ~/so101_lab/station.env`.
"""
import os
from pathlib import Path

import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

STEPS = 500
BATCH_SIZE = 8
LOG_EVERY = 50
OUT_DIR = Path("outputs/act_training_loop_demo")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = os.environ["DATASET"]

    # 1. Inputs and outputs come from the dataset's features: images + state in, action out.
    meta = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features)  # chunk_size=100
    policy = ACTPolicy(cfg).to(device)
    policy.train()
    preprocess, postprocess = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    # 2. Each sample needs the NEXT 100 actions as its label: ask the dataset for them by timestamp.
    delta_timestamps = {"action": [i / meta.fps for i in cfg.action_delta_indices]}  # 0, 1/30, ..., 99/30 s
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())  # AdamW, lr 1e-5
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=device.type != "cpu", drop_last=True
    )

    # 3. The loop: normalise, forward (L1 + 10 x KL), backward, step.
    step = 0
    while step < STEPS:
        for batch in loader:
            batch = preprocess(batch)
            loss, info = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % LOG_EVERY == 0:
                print(f"step {step:5d}  loss {loss.item():.3f}  l1 {info['l1_loss']:.3f}  kld {info.get('kld_loss', 0):.3f}")
            step += 1
            if step >= STEPS:
                break

    # 4. Save the weights together with the normalisation pre/post-processors.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(OUT_DIR)
    preprocess.save_pretrained(OUT_DIR)
    postprocess.save_pretrained(OUT_DIR)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
```

### A5. `act_inference_loop.py`: the ACT control loop on the robot

`policy.select_action` returns one action per call but only runs the network when its 100-action queue is empty, which is the chunk execution drawn in the concept section. Run with `python ~/so101_lab/act_inference_loop.py`; to test a local checkpoint, first `export POLICY_PATH=outputs/train/act_so101_cube_lift_${STATION}/checkpoints/last/pretrained_model`.

```python
"""act_inference_loop.py -- the core of lerobot-rollout for ACT on an SO-101 (LeRobot v0.6.1).

Adapted from LeRobot's examples/tutorial/act/act_using_example.py for the SO-101 and this lab's cameras.
Run after `source ~/so101_lab/station.env`. Camera names and sizes MUST match the training dataset.
"""
import os
import time

import torch
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
EPISODE_S = 20
N_EPISODES = 3


def cam(value: str):
    """Camera index (e.g. '0') or a device path (e.g. '/dev/v4l/by-id/...')."""
    return int(value) if value.isdigit() else value


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = ACTPolicy.from_pretrained(os.environ.get("POLICY_PATH", os.environ["POLICY"]))
    policy.to(device)
    policy.eval()

    # Dataset metadata gives the feature names and the normalisation statistics used in training.
    meta = LeRobotDatasetMetadata(os.environ["DATASET"])
    preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=meta.stats)

    cameras = {
        "front": OpenCVCameraConfig(index_or_path=cam(os.environ["FRONT_CAM"]), width=640, height=480, fps=FPS),
        "wrist": OpenCVCameraConfig(index_or_path=cam(os.environ["WRIST_CAM"]), width=640, height=480, fps=FPS),
    }
    robot = SO101Follower(
        SO101FollowerConfig(port=os.environ["FOLLOWER_PORT"], id=os.environ["FOLLOWER_ID"], cameras=cameras)
    )
    robot.connect()

    try:
        for ep in range(N_EPISODES):
            input(f"Episode {ep + 1}/{N_EPISODES}: put the cube on a spot, clear the area, press Enter ")
            policy.reset()  # empty the action queue so the first step plans a fresh 100-step chunk
            for _ in range(EPISODE_S * FPS):
                t0 = time.perf_counter()
                obs = robot.get_observation()  # 6 joint positions + one image per camera
                frame = build_inference_frame(
                    observation=obs, ds_features=meta.features, device=device, task=os.environ.get("TASK")
                )
                with torch.inference_mode():
                    # Runs the network only when the queue is empty (every 100 steps by default);
                    # otherwise it just pops the next queued action.
                    action = policy.select_action(preprocess(frame))
                robot.send_action(make_robot_action(postprocess(action), meta.features))
                precise_sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
```

## Instructor notes and assessment

The lab succeeds or fails on preparation: a pre-installed PC image, labelled hardware and one complete instructor run-through remove most of the friction in Session A.

### Before the first class

- [ ] Run the whole guide once on one station, Steps 1 to 7, and keep that dataset and policy as a **reference pair**
- [ ] Install the software on every lab PC (Software setup steps 1 to 7) and confirm CUDA prints `True`
- [ ] Label all boards, supplies and cables LEADER / FOLLOWER and number each station
- [ ] Pre-fill each station's `station.env` with `STATION` and copy the appendix scripts into `~/so101_lab/`
- [ ] Check each kit: if motor IDs arrived pre-set, Session A can skip 1b and 1c and spend the time on teleoperation practice
- [ ] Decide where training runs (lab PCs overnight, or Colab) and test it once
- [ ] Tape the 5 cube spots and the camera positions on every station

### Suggested Session A timing (3 h)

| Time | Activity |
| --- | --- |
| 0:00 to 0:30 | Safety briefing, concept overview (first two diagrams) |
| 0:30 to 1:00 | Hardware check, wiring, pre-flight checklist |
| 1:00 to 1:30 | Station file, find ports (Step 1a) |
| 1:30 to 2:15 | Motor IDs for both arms, if needed (Steps 1b, 1c) |
| 2:15 to 3:00 | Calibrate both arms, back up the calibration files (Step 2) |

### Fallbacks that keep a team moving

- **Dataset fails QA late in Session B:** the team trains on the instructor's reference dataset and records its own again later.
- **Training did not finish:** evaluate the newest checkpoint in `checkpoints/`, or the instructor's reference policy, so Session D still happens.
- **One arm faulty:** two teams share a station, alternating teleoperation and scoring.

### Misconceptions to address

- "Low training loss means it works." Only the real-robot success rate shows that (Step 7).
- "More episodes always help." Inconsistent episodes can hurt; 50 clean ones beat 100 messy ones.
- "Replay and the policy are the same thing." Replay is blind; the policy looks at the cameras every chunk (Step 5d).
- "Calibration is a one-off chore." Recalibrating after recording changes the meaning of the data.

### Assessment checklist (mapped to the lab learning outcomes)

| Lab outcome | Evidence the team submits | Meets the standard when |
| --- | --- | --- |
| 1. Set up and calibrate | Both calibration JSON files; short video of teleoperation | Follower tracks the leader smoothly through all joints |
| 2. Explain IL and ACT | Write-up paragraph, or a short viva using the concept diagrams | Explains chunking, the role of z, and why z = 0 at inference |
| 3. Record a clean dataset | Dataset id; output of `check_dataset.py`; logbook | 50 episodes, both cameras, no unexplained flagged episodes |
| 4. Train ACT | Training command; log excerpt or W&B chart | Run completes; loss falls and flattens; checkpoint on the Hub |
| 5. Evaluate and analyse | Results table; evaluation dataset id; answers to Step 7d | Success rate reported per spot; failures classified with evidence and a proposed data fix |

These map onto the unit's dataset-generation, policy-learning and policy-evaluation outcomes; the unit's VLA outcome is outside this lab.

### Intentional omissions

- **Arm assembly and 3D printing.** Assumed done; see the [SO-ARM100 repository](https://github.com/TheRobotStudio/SO-ARM100) and the [LeRobot SO-101 page](https://huggingface.co/docs/lerobot/so101).
- **Simulation.** No Isaac Sim / Isaac Lab or LeIsaac path here; this guide is real-arm only.
- **Other policies.** Diffusion Policy, SmolVLA and pi0 use the same dataset and similar commands but are not covered.
- **Human-in-the-loop correction (DAgger).** `lerobot-rollout --strategy.type=dagger` is a natural extension after Step 7, not part of this lab.
- **Hyperparameter tuning.** Only defaults are used, apart from optional temporal ensembling at inference.
- **Language conditioning and multiple tasks.** One task sentence throughout.
- **Depth cameras, macOS specifics, Hugging Face Jobs billing.** Mentioned at most in passing.
- **Verification status.** Commands, flags and APIs were checked against the LeRobot v0.6.1 source and documentation, and the helper scripts were syntax-checked; they have not been run on a physical SO-101 rig as part of writing this guide. Do the instructor run-through before class.

## References

Commands and defaults in this guide follow LeRobot v0.6.1 (source tag `v0.6.1`, checked November 2026).

- [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101): motor gearing, port finding, motor setup, calibration
- [LeRobot: imitation learning on real-world robots](https://huggingface.co/docs/lerobot/il_robots): teleoperation, recording, keyboard controls, training, resuming
- [LeRobot installation](https://huggingface.co/docs/lerobot/installation): Python 3.12, ffmpeg, extras, OS notes
- [LeRobot ACT policy page](https://huggingface.co/docs/lerobot/act): ACT architecture, training tips, first-policy recommendation
- [LeRobot policy deployment (lerobot-rollout)](https://huggingface.co/docs/lerobot/inference.md): base and episodic strategies, keyboard controls
- [LeRobot notebooks](https://huggingface.co/docs/lerobot/notebooks) and the [ACT training Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb)
- [LeRobot source code (GitHub)](https://github.com/huggingface/lerobot): `ACTConfig` defaults, `SOFollowerConfig`, dataset format v3.0, `examples/tutorial/act/`
- [SO-ARM100 / SO-101 hardware repository](https://github.com/TheRobotStudio/SO-ARM100): bill of materials, servo variants, power supplies, camera mounts
- [LeRobot dataset visualiser](https://huggingface.co/spaces/lerobot/visualize_dataset)
- Zhao, Kumar, Levine and Finn, [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705), arXiv:2304.13705, 2023: the ACT paper
