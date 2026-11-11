# CR10 Reinforcement Learning Lab

**MuJoCo · Gymnasium · PPO · Dobot CR10 + Robotiq 2F-85**  
Tested setup: **Ubuntu 22.04 / Python 3.10**. Internet is needed for the initial installation. ROS 2 is **not** needed for Stages 01–03.

## 1. Download & set up

Download **`CR10_RL_LAB_STUDENT_CLEAN.zip`** into `~/Downloads`, then run:

```bash
cd ~/Downloads
unzip -q CR10_RL_LAB_STUDENT_CLEAN.zip
cd CR10_NEW_PLATE_RL_FULL_CLASS_PACKAGE
bash setup.sh
```

> Run the commands below from `CR10_NEW_PLATE_RL_FULL_CLASS_PACKAGE`.
> Use `./run.sh` rather than system `python3`: the launcher selects the project's Python environment.

## 2. MuJoCo: first simulation (no RL)

```bash
./run.sh intro
```



## 3. CR10 + gripper (no RL)

```bash
./run.sh arm
```



## 4. Replay a pretrained PPO policy

```bash
./run.sh checkpoint
./run.sh replay-red
./run.sh replay-blue
```



## 5. Train PPO

Short demonstration (saves a *new* model; does not overwrite the pretrained one):

```bash
./run.sh train-quick
```

Outputs are saved in `02_Gripper_PPO_Training/artifacts/`:

```text
student_training_quick.zip
student_training_quick_vecnormalize.pkl
student_training_quick_metadata.json
```



```bash
./run.sh replay-red \
  --model artifacts/student_training_quick.zip \
  --normalizer artifacts/student_training_quick_vecnormalize.pkl
```

For a longer training experiment:

```bash
./run.sh train-full
```

## 6. Read the RL implementation

```bash
cd 02_Gripper_PPO_Training
```

| File | What to inspect |
|---|---|
| `train_color_sb3.py` | PPO configuration and training loop |
| `replay_color_policy.py` | Loading a policy and running inference |
| `evaluate_color_policy.py` | Evaluating success across episodes |
| `cr10_2f85_lab/` | Robot environment, observations, actions, and rewards |
| `artifacts/` | Saved policy and observation normalization |

Return to the main lab folder with `cd ..`.

## 7. How the physical CR10 is connected 


```bash
./run.sh physical-code
```


**D405 camera → color/depth perception → PPO observation → policy action → safety/controller checks → ROS 2 / Dobot commands.**

Stage 04 uses a **plastic pushing plate**, not the simulated 2F-85 gripper. Its deployment policy and calibration are **not interchangeable** with the gripper simulation checkpoint. The physical source is provided to study the real implementation.
## Quick command reference

```bash
./run.sh intro          # MuJoCo introduction
./run.sh arm            # CR10 without RL
./run.sh checkpoint     # Check pretrained model
./run.sh replay-red     # Pretrained policy, red target
./run.sh replay-blue    # Pretrained policy, blue target
./run.sh evaluate       # Evaluate pretrained policy
./run.sh train-quick    # Train briefly (separate checkpoint)
./run.sh train-full     # Longer training, optional
./run.sh physical-code  # List real-robot implementation files
```


