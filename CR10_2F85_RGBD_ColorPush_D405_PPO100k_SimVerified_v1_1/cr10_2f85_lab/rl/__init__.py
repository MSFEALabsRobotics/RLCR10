from __future__ import annotations

from gymnasium.envs.registration import register, registry


ENV_ID = "CR10-2F85-TabletopPush-v0"
TRAINING_ENV_ID = "CR10-2F85-TabletopPushForwardOnly-v0"
COLOR_TARGET_ENV_ID = "CR10-2F85-RGBDColorTargetPush-v0"


def register_env() -> None:
    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point=(
                "cr10_2f85_lab.rl.tabletop_push_env:TabletopPushEnv"
            ),
        )
    if TRAINING_ENV_ID not in registry:
        register(
            id=TRAINING_ENV_ID,
            entry_point=(
                "cr10_2f85_lab.rl.fixed_grip_push_env:FixedGripPushEnv"
            ),
        )
    if COLOR_TARGET_ENV_ID not in registry:
        register(
            id=COLOR_TARGET_ENV_ID,
            entry_point=(
                "cr10_2f85_lab.rl.color_target_push_env:ColorTargetPushEnv"
            ),
        )


register_env()

from .tabletop_push_env import OBSERVATION_COMPONENTS, TabletopPushEnv
from .fixed_grip_push_env import (
    POLICY_ACTION_SCHEMA_VERSION,
    FixedGripPushEnv,
)
from .color_target_push_env import (
    COLOR_OBSERVATION_COMPONENTS,
    ColorTargetPushEnv,
    scripted_color_action,
)

__all__ = [
    "ENV_ID",
    "TRAINING_ENV_ID",
    "COLOR_TARGET_ENV_ID",
    "OBSERVATION_COMPONENTS",
    "POLICY_ACTION_SCHEMA_VERSION",
    "TabletopPushEnv",
    "FixedGripPushEnv",
    "COLOR_OBSERVATION_COMPONENTS",
    "ColorTargetPushEnv",
    "scripted_color_action",
    "register_env",
]
