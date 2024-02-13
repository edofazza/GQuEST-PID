import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame

from src.rp.rppid import RedPitayaPID


class RedPitayaEnv(gym.Env):
    def __init__(self, hostname: str = '169.254.167.128'):
        self.rp = RedPitayaPID(hostname)
        self.action_space = spaces.Box(low=0, high=3, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.reward_range = (-1, 0)
        self.setpoint = 0.32

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # Reset Red Pitaya
        self.rp.reset()
        # set temperature
        self.rp.ramp_piezo()
        if not self.rp.scan_temperature(500):
            print('ERR: could not find TEM00 mode, resetting')
            self.reset()
        # lock
        time.sleep(10)
        try:
            self.rp.lock_cavity()
        except:
            print('ERR: could not lock the cavity, resetting')
            self.reset()
        # get the setpoint
        self.setpoint = self.rp.redpitaya.pid0.setpoint
        # get state
        print('Setpoint: ', self.setpoint)
        _, in1 = self.rp.scope(input2='iq0')
        return in1.mean(), {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # change PID parameters
        self.rp.set_pid0(action[0], action[1], action[2])
        time.sleep(0.1)  # TODO: is it too much?
        # get state
        in1, phcav = self.rp.scope(input1='iq0')
        next_state = in1.mean()
        reward = -abs(next_state - self.setpoint)
        done = False
        if phcav.max() < 0.95:
            done = True
        return next_state, reward, done, False, {}  # next_obs, reward, terminated (bool), truncated (bool), info (dict)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None

    def close(self):
        self.rp.reset()