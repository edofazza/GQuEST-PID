import numpy as np
from rpcontrol import RedPitayaController


class RedPitayaScope(RedPitayaController):
    def __init__(self, hostname: str, user: str = 'root', password: str = 'root', config: str = 'fermi',
                 gui: bool = False):
        super().__init__(hostname, user, password, config, gui)

    def scope(self, input1: str = 'out1', input2: str = 'in2', hysteresis: float = 0.01,
              trigger_source: str = 'immediately', ordered: bool = False):
        self.redpitaya.scope.decimation = 256
        self.redpitaya.scope.input1 = input1
        # Scope's second input
        self.redpitaya.scope.input2 = input2
        self.redpitaya.scope.threshold = self.redpitaya.asg0.offset + self.redpitaya.asg0.amplitude / 2
        # Trigger Hysteresis
        self.redpitaya.scope.hysteresis = hysteresis
        # Trigger Source
        self.redpitaya.scope.trigger_source = trigger_source
        # Trigger Time Delay
        self.redpitaya.scope.trigger_delay = 0
        # Take a Scope Trace
        purple_signal, blue_signal = self.redpitaya.scope.single()
        if ordered:
            purple_signal_peak_index = np.where(purple_signal == purple_signal.max())[0][0]
            first_position = purple_signal_peak_index + 1
            purple_signal = np.concatenate((purple_signal[first_position:], purple_signal[:first_position]))
            blue_signal = np.concatenate((blue_signal[first_position:], blue_signal[:first_position]))
        return purple_signal, blue_signal
