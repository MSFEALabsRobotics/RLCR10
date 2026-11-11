"""The user's plate-trained PPO with original saved VecNormalize statistics."""
from pathlib import Path
import numpy as np
from .core import OBS_LOW, OBS_HIGH, SafetyFault

class Policy:
    def __init__(self, root):
        import gymnasium as gym
        from gymnasium.spaces import Box
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        class ObservationOnlyEnv(gym.Env):
            observation_space = Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
            action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(12, dtype=np.float32), {}
            def step(self, action):
                raise RuntimeError('ObservationOnlyEnv is for normalization only')

        root = Path(root)
        self.ve = DummyVecEnv([lambda: ObservationOnlyEnv()])
        self.normalizer = VecNormalize.load(
            str(root/'artifacts/plate_color_ppo_vecnormalize.pkl'), self.ve
        )
        self.normalizer.training = False
        self.normalizer.norm_reward = False
        self.network = PPO.load(str(root/'artifacts/plate_color_ppo.zip'), device='cpu')
        for label, space in (
            ('model observations', self.network.observation_space),
            ('normalizer observations', self.normalizer.observation_space),
        ):
            if tuple(space.shape) != (12,) or not np.allclose(space.low, OBS_LOW) or not np.allclose(space.high, OBS_HIGH):
                raise SafetyFault(f'{label} do not match the plate-training observation space')
        if tuple(self.network.action_space.shape) != (2,):
            raise SafetyFault('PPO action space is not two-dimensional')
        if not np.allclose(self.network.action_space.low, [-1.,-1.]) or not np.allclose(self.network.action_space.high, [1.,1.]):
            raise SafetyFault('PPO action range does not match [-1,1]')

    def predict(self, raw_obs):
        obs = np.asarray(raw_obs, dtype=np.float32)
        if obs.shape != (12,) or not np.all(np.isfinite(obs)):
            raise SafetyFault('policy requires one finite raw 12D observation')
        if np.any(obs < OBS_LOW - 1e-5) or np.any(obs > OBS_HIGH + 1e-5):
            raise SafetyFault('raw observation outside training bounds')
        # Never normalize inputs twice. This is exactly the evaluation-time step.
        normalized = self.normalizer.normalize_obs(obs.copy())
        action, _ = self.network.predict(normalized, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(action)) or np.any(np.abs(action)>1.00001):
            raise SafetyFault('PPO returned invalid action')
        return action

    def close(self):
        self.normalizer.close()
