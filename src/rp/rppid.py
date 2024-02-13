import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from rpscope import RedPitayaScope


class RedPitayaPID(RedPitayaScope):
    def __init__(self, hostname: str, user: str = 'root', password: str = 'root', config: str = 'fermi',
                 gui: bool = False):
        super().__init__(hostname, user, password, config, gui)

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

        print("Close the feedback loop")
        # Close the Feedback Loop
        # Set PID gains and corner frequencies
        # Set Point
        self.redpitaya.pid0.setpoint = poptLine[1]
        print('Setpoint ', poptLine[1])
        self.set_pid0()

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

    def loop_auto_lock(self):
        self.reset()
        self.ramp_piezo()
        if self.scan_temperature(500):
            time.sleep(10)
            self.lock_cavity()
            starting_time = time.time()
            print(f'Locked at: {starting_time}')
            while True:
                time.sleep(10)
                print(f'Seconds after start: {time.time() - starting_time}')
                purple_signal, blue_signal = self.scope()
                print(f'purple (fast) signal mean: {purple_signal.mean()}')
                print(f'blue signal mean: {blue_signal.mean()}')
                if blue_signal.max() < 0.95:
                    end_time = time.time()
                    print(f'Lost lock at: {end_time}. Took {end_time-starting_time}')
                    break
        self.loop_auto_lock()

    def lock_and_reset(self):
        self.reset()
        self.ramp_piezo()
        if self.scan_temperature(1000):
            time.sleep(10)
            self.lock_cavity()
            time.sleep(1)
            in1, _ = self.scope(input1='iq0')
            print(in1.min(), in1.mean(), in1.max())
            time.sleep(1)
            _, in1 = self.scope(input2='iq0')
            print(in1.min(), in1.mean(), in1.max())
            _, in1 = self.scope(input2='iq0')
            print(in1.min(), in1.mean(), in1.max())
            time.sleep(1)
            _, in1 = self.scope(input2='iq0')
            print(in1.min(), in1.mean(), in1.max())
        else:
            print('TEMP non found')
        self.reset()


if __name__ == '__main__':
    rdpid = RedPitayaPID('169.254.167.128')
    rdpid.loop_auto_lock()
