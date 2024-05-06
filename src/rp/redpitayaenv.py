import time
from typing import Any, SupportsFloat, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
import scipy

from rpscope import RedPitayaScope


class RedPitayaEnv(gym.Env):
    def __init__(self, rp):
        self.rp = rp
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.reward_range = (-0.95, 0.04)
        self.reset()

    def scan_piezo(self, asg: bool = True, output_direct: str = 'out1', amp: float = 0.5,
                  offset: float = 0.5, freq: float = 1e2) -> None:
        if asg:
            self.rp.set_asg0(waveform='halframp', output_direct=output_direct, amp=amp, offset=offset, freq=freq)
        else:
            self.rp.set_asg1(waveform='halframp', output_direct=output_direct, amp=amp, offset=offset, freq=freq)

    def ramp_piezo(self, phase=15):
        self.reset()
        self.scan_piezo(freq=1 / (8E-9 * (2 ** 14) * 256))
        self.rp.set_iq0(phase=phase)

    def scan_temperature(self, epsilon=1000) -> bool:
        for i in np.arange(0, 0.3, 0.00025):
            # set temperature
            self.set_dac2(i)
            # take scope
            _, blue_signal = self.scope(ordered=True)
            half_scope_trace = int(blue_signal.shape[0]/2)

            blue_signal_peak_index = np.where(blue_signal == blue_signal.max())[0][0]
            if half_scope_trace - epsilon < blue_signal_peak_index < half_scope_trace + epsilon and \
                    blue_signal.max() > .95:
                print('Temperature: :', i)
                return True
        return False

    def lock_cavity(self, phase=20):
        #####
        #   RAMP PIEZO
        print("Scan Piezo")
        self.scan_piezo(freq=1 / (8E-9 * (2 ** 14) * 256))
        print("Run on Modulation")
        self.rp.set_iq0(phase=phase)
        ######
        print("Take a scope trace")
        scope_trace = self.rp.scope('out1', 'iq0', trigger_source='ch1_positive_edge')
        print("Done taking scope trace")
        np.save('scope_trace.npy', scope_trace)
        print("Saved scope trace")
        ch1, ch2 = scope_trace

        # Guess Initial Parameters
        print("Curve fit")
        offs = np.mean(ch2)
        gamma = (np.max(ch1) - np.min(ch1)) / 10
        x0 = (np.max(ch1) - np.min(ch1)) / 2
        amp = (np.max(ch2) - np.mean(ch2)) * (x0 ** 3)  # guessed,but don't guess the correct sign

        # Curve Fit
        poptLine, pcovLine = scipy.optimize.curve_fit(self.lorantian_derivative, ch1, ch2, p0=[amp, offs, gamma, x0])
        fit = self.lorantian_derivative(ch1, poptLine[0], poptLine[1], poptLine[2], poptLine[3])

        print("Go back to resonance")
        # Go to resonance (CONSTANT PIEZO)
        # self.constantPzt(V=poptLine[3])
        self.rp.set_asg0(waveform='dc', output_direct='out1', offset=poptLine[3])


    @staticmethod
    def lorantian_derivative(x, A, B, g, x0):  # derivative a Lorantian
        return -2 * A * (x - x0) / (((x - x0) ** 2 + g ** 2) ** 2) + B

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # Reset Red Pitaya
        self.rp.reset()
        # set temperature
        self.ramp_piezo()
        if not self.scan_temperature(500):
            print('ERR: could not find TEM00 mode, resetting')
            self.reset()
        # lock
        time.sleep(10)
        try:
            self.lock_cavity()
        except:
            print('ERR: could not lock the cavity, resetting')
            self.reset()
        _, phcav = self.rp.scope()
        return phcav.max(), {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        time.sleep(0.001)
        # change temperature
        self.rp.set_dac2(self.rp.redpitaya.ams.dac2 + action)
        time.sleep(0.0001)  # TODO: is it too much?
        # get state
        _, phcav = self.rp.scope()
        next_state = phcav.max()
        reward = next_state - 0.95
        done = False
        if phcav.max() < 0.95:
            done = True
        return next_state, reward, done, False, {}  # next_obs, reward, terminated (bool), truncated (bool), info (dict)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None

    def close(self):
        self.rp.reset()