from abc import ABC, abstractmethod
import pyrpl


class RedPitaya(ABC):
    @abstractmethod
    def __init__(self, hostname: str, user: str = 'root', password: str = 'root', config: str = 'fermi',
                 gui: bool = False):
        try:
            p = pyrpl.RedPitaya()
            self.redpitaya = p  # Access the RedPitaya object in charge of communicating with the board
        except Exception as e:
            print(e)

    @abstractmethod
    def reset(self) -> None:
        pass


    @abstractmethod
    def set_asg0(self, waveform: str = 'halframp', output_direct: str = 'out1', amp: float = 0.5,
                 offset: float = 0.5, freq: float = 1e2) -> None:
        pass

    @abstractmethod
    def set_asg1(self, waveform: str = 'halframp', output_direct: str = 'out1', amp: float = 0.5,
                 offset: float = 0.5, freq: float = 1e2) -> None:
        pass

    @abstractmethod
    def set_iq0(self, frequency: float = 25e6, bandwidth: list = [2e6, 2e6], gain: float = 0.5, phase: int = 0,
                acbandwidth: float = 5e6, amplitude: float = 1., input: str = 'in1', output_direct: str = 'out2',
                output_signal: str = 'quadrature', quadrature_factor: int = 1) -> None:
        pass

    @abstractmethod
    def set_dac2(self, voltage: float = 0.) -> None:
        pass

    @abstractmethod
    def set_pid0(self, ival: float = 0, integrator: float = 1e3, proportional: float = 0,
                 differantiator: float = 0, input='iq0', output_direct: str = 'out1') -> None:
        pass

