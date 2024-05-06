from redpitaya import RedPitaya


class RedPitayaController(RedPitaya):
    def __init__(self, hostname: str, user: str = 'root', password: str = 'root', config: str = 'fermi',
                 gui: bool = False):
        super().__init__(hostname, user, password, config, gui)

    def reset(self) -> None:
        # Turn off arbitrary signal generator channel 0
        self.set_asg0(output_direct='off', amp=0, offset=0)
        # Turn off arbitrary signal generator channel 1
        self.set_asg1(output_direct='off', amp=0, offset=0)
        # Turn off I+Q quadrature demodulation/modulation modules
        self.redpitaya.iq0.output = 'off'
        self.set_iq0(output_direct='off')
        # Turn off PID module 0
        self.set_pid0(0, 0, 0, 0, 'off')
        # Turn off dac2
        self.set_dac2(0)

    def set_asg0(self, waveform: str = 'halframp', output_direct: str = 'out1', amp: float = 0.5,
                 offset: float = 0.5, freq: float = 1e2) -> None:
        self.redpitaya.asg0.setup(waveform=waveform, output_direct=output_direct, trigger_source='immediately',
                                  offset=offset, amplitude=amp, frequency=freq)

    def set_asg1(self, waveform: str = 'halframp', output_direct: str = 'out1', amp: float = 0.5,
                 offset: float = 0.5, freq: float = 1e2) -> None:
        self.redpitaya.asg1.setup(waveform=waveform, output_direct=output_direct, trigger_source='immediately',
                                  offset=offset, amplitude=amp, frequency=freq)

    def set_iq0(self, frequency: float = 25e6, bandwidth: list = [2e6, 2e6], gain: float = 0.5, phase: int = 0,
                acbandwidth: float = 5e6, amplitude: float = 1., input: str = 'in1', output_direct: str = 'out2',
                output_signal: str = 'quadrature', quadrature_factor: int = 1) -> None:
        self.redpitaya.iq0.setup(frequency=frequency, bandwidth=bandwidth, gain=gain, phase=phase,
                                 acbandwidth=acbandwidth, amplitude=amplitude, input=input, output_direct=output_direct,
                                 output_signal=output_signal, quadrature_factor=quadrature_factor)

    def set_dac2(self, voltage: float = 0.) -> None:
        self.redpitaya.ams.dac2 = voltage   # pin 17 output 0

    def set_pid0(self, ival: float = 0, integrator: float = 1e3, proportional: float = 0,
                 differantiator: float = 0, input='iq0', output_direct: str = 'out1') -> None:
        # Clear integrator
        self.redpitaya.pid0.ival = ival
        # Proportinal
        self.redpitaya.pid0.p = proportional
        # Integrator
        self.redpitaya.pid0.i = integrator
        # differentiator
        self.redpitaya.pid0.d = differantiator
        # input or output
        self.redpitaya.pid0.input = input
        self.redpitaya.pid0.output_direct = output_direct


