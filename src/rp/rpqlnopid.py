import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from rpscope import RedPitayaScope


def round_to_nearest_0_1(value):
    return round(value * 10) / 10


def round_to_nearest_0_05(value):
    return round(value * 20) / 20


class RedPitayaQLearningNoPID(RedPitayaScope):
    def __init__(self, hostname: str, user: str = 'root', password: str = 'root', config: str = 'fermi',
                 gui: bool = False,  load=False, learning_rate=0.4, discout_factor=0.99, epsilon=0.7,
                 num_episodes=5000, test=False):
        super().__init__(hostname, user, password, config, gui)

        self.voltage_range = np.arange(-1, 1.1, 0.1)
        self.num_states = len(self.voltage_range)  # 21
        # actions
        self.action_range = np.arange(-.001, .0015, .0005)
        self.num_actions = len(self.action_range)  # 5
        # Q-Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discout_factor
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.test = test

        if self.test:
            self.epsilon = 0

        # Q-values
        if load:
            print('Loaded Q-Matrix')
            self.Q = np.load('q_qlearning.npy')
        else:
            self.Q = np.zeros((self.num_states, self.num_actions))

    def scan_piezo(self, asg: bool = True, output_direct: str = 'out1', amp: float = 0.5,
                  offset: float = 0.5, freq: float = 1e2) -> None:
        if asg:
            self.set_asg0(waveform='halframp', output_direct=output_direct, amp=amp, offset=offset, freq=freq)
        else:
            self.set_asg1(waveform='halframp', output_direct=output_direct, amp=amp, offset=offset, freq=freq)

    def ramp_piezo(self, phase=15):
        self.reset()
        self.scan_piezo(freq=1 / (8E-9 * (2 ** 14) * 256))
        self.set_iq0(phase=phase)

    def lock_cavity(self, phase=20):
        #####
        #   RAMP PIEZO
        print("Scan Piezo")
        self.scan_piezo(freq=1 / (8E-9 * (2 ** 14) * 256))
        print("Run on Modulation")
        self.set_iq0(phase=phase)
        ######
        print("Take a scope trace")
        scope_trace = self.scope('out1', 'iq0', trigger_source='ch1_positive_edge')
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
        print("Plot")
        # Plot Measured Data and Curve Fit
        plt.figure(1)
        plt.title("PDH Error Signal")
        plt.xlabel("PZT Drive Voltage (V)")
        plt.ylabel("Error Signal (V)")
        plt.grid(True)
        plt.plot(ch1, ch2)
        plt.plot(ch1, fit)

        print("Go back to resonance")
        # Go to resonance (CONSTANT PIEZO)
        # self.constantPzt(V=poptLine[3])
        self.set_asg0(waveform='dc', output_direct='out1', offset=poptLine[3])

        """print("Close the feedback loop")
        # Close the Feedback Loop
        # Set PID gains and corner frequencies
        # Set Point
        self.redpitaya.pid0.setpoint = poptLine[1]
        print('Setpoint ', poptLine[1])
        self.set_pid0()"""

    @staticmethod
    def lorantian_derivative(x, A, B, g, x0):  # derivative a Lorantian
        return -2 * A * (x - x0) / (((x - x0) ** 2 + g ** 2) ** 2) + B

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

    def _get_state_index(self, temperature):
        return np.argmin(np.abs(self.voltage_range - temperature))

    def _get_action_index(self, action):
        return np.argmin(np.abs(self.action_range - action))

    def qlearning(self, episode: int = 0):
        while episode < self.num_episodes:
            print(f'EPISODE {episode}')
            if episode == 1000 and not self.test:
                self.epsilon = 0.3
            self.reset()
            self.ramp_piezo()
            if self.scan_temperature(500):
                time.sleep(10)
                try:
                    self.lock_cavity()
                except:
                    continue
                system_unlock = False
                print(f'\tLocked at: {time.time()}')
                print(f'\tInitial temperature voltage: {self.redpitaya.ams.dac2}V')
                purple_signal, _ = self.scope()
                print(f'\tFast signal mean: {purple_signal.max()}')
                state = self._get_state_index(round_to_nearest_0_1(purple_signal.max()))
                while True:
                    time.sleep(0.001)   # TODO: 1 0.1
                    # Choose action using epsilon-greedy policy
                    if np.random.rand() < self.epsilon and not self.test:
                        action = np.random.choice(self.num_actions)
                    else:
                        action = np.argmax(self.Q[state, :])
                    print(f'\tAction index: {action}')
                    self.set_dac2(self.redpitaya.ams.dac2 + self.action_range[action])
                    print(f'\tTemperature voltage: {self.redpitaya.ams.dac2}V')
                    time.sleep(0.0001)    # TODO: 0.1 0.01
                    # Get the next state, reward, and system_unlock
                    purple_signal, blue_signal = self.scope()
                    print(f'\tFast signal mean: {purple_signal.max()}')
                    if blue_signal.max() < 0.95:
                        print(f'\tLost lock at: {time.time()}')
                        system_unlock = True
                    next_state = self._get_state_index(round_to_nearest_0_1(purple_signal.max()))
                    reward = 1 if not system_unlock else 0
                    print(f'\tState index: {next_state}')
                    # Update Q-values
                    if not self.test:
                        max_next_q = np.max(self.Q[next_state, :])
                        self.Q[state, action] = (self.Q[state, action] +
                                                 self.learning_rate * (reward +
                                                                       self.discount_factor * max_next_q -
                                                                       self.Q[state, action]))
                    state = next_state

                    if system_unlock:
                        if not self.test:
                            np.save('q_qlearning.npy', self.Q)
                        episode += 1
                        break


if __name__ == '__main__':
    rpql = RedPitayaQLearningNoPID('169.254.167.128')
    rpql.qlearning()
