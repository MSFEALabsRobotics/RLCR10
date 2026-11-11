#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
echo '[SETUP] Stage 01 MuJoCo model and Python environment'
bash 01_MuJoCo_CR10_No_RL/setup.sh
PY="$(pwd)/01_MuJoCo_CR10_No_RL/.venv/bin/python"
[[ -f 02_Gripper_PPO_Training/artifacts/pack5_v02_color_ppo_100k_seed7.zip ]] || { echo 'Gripper checkpoint missing' >&2; exit 2; }
[[ -f 02_Gripper_PPO_Training/artifacts/pack5_v02_color_ppo_100k_seed7_vecnormalize.pkl ]] || { echo 'Gripper normalizer missing' >&2; exit 2; }
"$PY" -m pip check
./run.sh checkpoint
printf '\nREADY: ./run.sh intro ; ./run.sh arm ; ./run.sh replay-red\n'
