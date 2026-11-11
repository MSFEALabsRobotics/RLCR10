#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
trap 'echo "[SETUP FAILED] Check the command above. See README.md troubleshooting." >&2' ERR
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v git >/dev/null || { echo "Install git: sudo apt install git"; exit 2; }
"$PYTHON_BIN" - <<'PY_CHECK'
import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    raise SystemExit(f"Expected Python 3.10 (Ubuntu 22.04), found {sys.version.split()[0]}. Try PYTHON_BIN=python3.10 bash setup.sh")
print("Python",sys.version.split()[0])
PY_CHECK
if [[ ! -d .venv ]]; then "$PYTHON_BIN" -m venv .venv; fi
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[rl]'

# Get only the locked model-description folders; ROS installation is not needed.
fetch_locked() {
    local name="$1" url="$2" revision="$3" branch="$4"; shift 4
    local location="vendor/$name"
    if [[ -d "$location/.git" ]]; then
        echo "Using existing source checkout: $location"
        local current; current="$(git -C "$location" rev-parse HEAD)"
        [[ "$current" == "$revision" ]] || { echo "Commit mismatch for $location: $current"; exit 3; }
        return
    fi
    echo "Downloading pinned $name model sources (requires internet)..."
    mkdir -p "$location"
    git init -q "$location"
    git -C "$location" remote add origin "$url"
    git -C "$location" sparse-checkout init --cone
    git -C "$location" sparse-checkout set "$@"
    git -C "$location" fetch --depth 1 origin "$revision"
    if [[ "$branch" == DETACHED ]]; then
        git -C "$location" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
    else
        git -C "$location" checkout -q -b "$branch" FETCH_HEAD
    fi
}
fetch_locked dobot     https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4.git     0f21e9839888f188ac06b0a9fea622d13a29a74f     feature/v4-optimization     dobot_rviz/urdf dobot_rviz/meshes/cr10
fetch_locked robotiq     https://github.com/robotiq/robotiq-ros.git     3ab3befccaa10468f80803ba687105c9d224d567     DETACHED     grippers/robotiq_description/urdf
fetch_locked menagerie     https://github.com/google-deepmind/mujoco_menagerie.git     da76818e269b82289eba39808e2fb91d679d6994     DETACHED     robotiq_2f85
python lab/prepare_models.py
python lab/check_ready.py --basic
printf '\n[READY] Installation and model construction completed.\n'
printf 'Try: ./run.sh arm\n'
printf 'No PPO model is installed in this stage. Use ../run.sh replay-red for the NEW plate policy.\n'
