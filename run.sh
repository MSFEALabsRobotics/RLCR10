#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/01_MuJoCo_CR10_No_RL/.venv/bin/python"
STAGE1="$ROOT/01_MuJoCo_CR10_No_RL"
SRC="$ROOT/02_Gripper_PPO_Training"
DOBOT="$STAGE1/vendor/dobot"
ROBOTIQ="$STAGE1/vendor/robotiq"
MENAGERIE="$STAGE1/vendor/menagerie"
[[ -x "$PY" ]] || { echo "Missing working Python: $PY" >&2; exit 2; }
cmd="${1:-help}"; if [[ "$#" -gt 0 ]]; then shift; fi
case "$cmd" in
intro)
 cd "$STAGE1"; exec "$PY" lab/00_mujoco_intro.py "$@" ;;
arm)
 cd "$STAGE1"; exec "$PY" lab/01_cr10_no_rl.py "$@" ;;
checkpoint)
 cd "$SRC"; exec "$PY" -c 'from pathlib import Path;from stable_baselines3 import PPO;p=Path("artifacts/pack5_v02_color_ppo_100k_seed7.zip");n=Path("artifacts/pack5_v02_color_ppo_100k_seed7_vecnormalize.pkl");assert p.is_file() and n.is_file(),"Checkpoint or VecNormalize missing";m=PPO.load(p,device="cpu");print("PASS: PPO model and normalizer present; observation:",m.observation_space,"action:",m.action_space)' ;;
replay-red|replay-blue)
 [[ -d "$DOBOT/.git" && -d "$ROBOTIQ/.git" && -d "$MENAGERIE/.git" ]] || { echo 'Robot sources missing. Stage 01 model setup incomplete.' >&2; exit 2; }
 if [[ "$cmd" == replay-red ]]; then color=red; sign=-1; else color=blue; sign=1; fi
 cd "$SRC"
 exec "$PY" replay_color_policy.py --target "$color" --red-lateral-sign "$sign" --seed 7 --viewer --dobot "$DOBOT" --robotiq "$ROBOTIQ" --menagerie "$MENAGERIE" "$@" ;;
evaluate)
 cd "$SRC"
 exec "$PY" evaluate_color_policy.py --episodes 8 --seed 7 --device cpu --dobot "$DOBOT" --robotiq "$ROBOTIQ" --menagerie "$MENAGERIE" "$@" ;;
train-quick)
 cd "$SRC"
 exec "$PY" train_color_sb3.py --timesteps 2048 --eval-episodes 2 --seed 7 --device cpu --headless --output artifacts/student_training_quick --dobot "$DOBOT" --robotiq "$ROBOTIQ" --menagerie "$MENAGERIE" "$@" ;;
train-full)
 cd "$SRC"
 exec "$PY" train_color_sb3.py --timesteps 100000 --eval-episodes 8 --seed 7 --device cpu --headless --output artifacts/student_training_full --dobot "$DOBOT" --robotiq "$ROBOTIQ" --menagerie "$MENAGERIE" "$@" ;;
physical-code)
 echo '[READ ONLY] Instructor physical plate hardware scripts; no motion is commanded.'
 if [[ -d "$ROOT/04_Real_CR10_Instructor_Only" ]]; then HW="$ROOT/04_Real_CR10_Instructor_Only"; else HW="$ROOT/04_Real_CR10"; fi
 find "$HW" -maxdepth 2 -type f \( -name '*.py' -o -name 'README*' \) | head -60 ;;
physical-status)
 echo '[READ ONLY] Physical plate stage is separate from the gripper simulation policy.'
 echo 'Gripper PPO is NOT a plate hardware deployment checkpoint.' ;;
help|*)
 cat <<'HELP'
Usage: ./run.sh COMMAND
  intro           MuJoCo introduction (no RL)
  arm             CR10 + gripper scripted demonstration (no RL)
  checkpoint      Check ORIGINAL gripper PPO model and VecNormalize
  replay-red      ORIGINAL gripper PPO pushes red cube in MuJoCo
  replay-blue     ORIGINAL gripper PPO pushes blue cube in MuJoCo
  evaluate        Evaluate frozen gripper policy (8 episodes)
  train-quick     2048-step PPO training walkthrough (separate output)
  train-full      100k-step PPO training (separate output)
  physical-code   List instructor-only real plate hardware code; NO movement
  physical-status Describe hardware separation

Stages 02/03 use the gripper simulation. Stage 04 is a separate plate project.
HELP
 ;;
esac
