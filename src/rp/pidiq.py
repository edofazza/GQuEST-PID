import time
import numpy as np
from rppid import RedPitayaPID


class PIDIQ(RedPitayaPID):
    def __init__(self, hostname: str):
        super().__init__(hostname)
        self.init_time = time.time()
        self.counter = 1

    def analyze(self):
        if self.init_time + 240 < time.time():
            return
        self.reset()
        self.ramp_piezo()
        if self.scan_temperature(500):
            time.sleep(10)
            try:
                self.lock_cavity()
            except:
                self.analyze()
            starting_time = time.time()
            print(f'Locked at: {starting_time}')
            while True:
                time.sleep(10)
                print(f'Seconds after start: {time.time() - starting_time}')
                out1, iq0 = self.scope(input2='asg0')
                _, in2 = self.scope()

                if in2.max() < 0.95:
                    end_time = time.time()
                    print(f'Lost lock at: {end_time}. Took {end_time-starting_time} ({self.counter})')
                    break
                else:
                    np.save(f'dataset/iq0/{self.counter}.npy', iq0)
                    if self.counter < 100:
                        np.save(f'dataset/out1/{self.counter}.npy', out1)
                    self.counter += 1
                    print(f'purple (fast) signal mean: {out1.mean()}')
                    print(f'blue signal mean: {in2.mean()}')
        self.analyze()


if __name__ == "__main__":
    pid = PIDIQ('169.254.167.128')
    pid.analyze()
