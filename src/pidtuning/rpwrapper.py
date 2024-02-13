import time
from typing import SupportsFloat, Any

from gymnasium import Wrapper
from gymnasium.core import WrapperActType, WrapperObsType


class SkipSteps(Wrapper):
    def __init__(self, env, skip=10):
        super().__init__(env)
        self._skip = skip

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0.
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            time.sleep(0.1)
            if done:
                break
        return obs, total_reward, done, trunk, info
