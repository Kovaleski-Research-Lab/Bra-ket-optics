import torch
import lightning as L


class Antenna(L.LightningModule):
    """
    Antenna class to represent an antenna in 3D space.
    Attributes:
        center (list[float]): Center coordinates of the antenna [x, y, z].
        wavelength (float): Wavelength of the antenna.
        gradients (str): Defines what parameters are optimizeable. Can be 
            'complex', 'amplitude_only', 'phase_only', or 'none' (default).
    """
    def __init__(self, center: list[float], 
                 wavelength: float,
                 gradients: str = 'none'):
        super().__init__()
        self.center = torch.tensor(center, dtype=torch.float32)
        self.wavelength = torch.tensor(wavelength)
        self.gradients = gradients
        self._initialize_transmissivity_function()

    def _initialize_transmissivity_function(self):
        self.amplitude = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), 
                                          requires_grad=self.gradients in ['complex', 'amplitude_only'])
        self.phase = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32),
                                          requires_grad=self.gradients in ['complex', 'phase_only'])
        self.transmissivity_function = self.amplitude * torch.exp(1j * self.phase)


    def set_amplitude(self, amplitude: float):
        with torch.no_grad():
            self.amplitude.fill_(amplitude)
            self.transmissivity_function = self.amplitude * torch.exp(1j * self.phase)

    def set_phase(self, phase: float):
        with torch.no_grad():
            self.phase.fill_(phase)
            self.transmissivity_function = self.amplitude * torch.exp(1j * self.phase)

    def set_transmissivity_function(self, tf: complex):
        with torch.no_grad():
            amplitude = tf.abs()
            phase = tf.angle()
            self.amplitude.fill_(amplitude)
            self.phase.fill_(phase)
            self.transmissivity_function = self.amplitude * torch.exp(1j * self.phase)

    def get_amplitude(self):
        return self.amplitude

    def get_phase(self):
        return self.phase

    def get_transmissivity_function(self):
            return self.transmissivity_function

    def get_position(self):
        return self.center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transmissivity_function * x




if __name__ == "__main__":
    center = [0.0, 0.0, 0.0]
    wavelength = 0.3
    antenna = Antenna(center=center, wavelength=wavelength, gradients='complex')
    print("Antenna Center:", antenna.get_position())
    print("Transmissivity Function:", antenna.get_transmissivity_function())
    antenna.set_amplitude(0.5)
    print("New transmissivity Function after setting amplitude to 0.5:", antenna.get_transmissivity_function())
    antenna.set_phase(torch.pi / 4)
    print("New transmissivity Function after setting phase to pi/4:", antenna.get_transmissivity_function())

    input_signal = torch.tensor([1.0 + 0.5j], dtype=torch.complex64)
    output_signal = antenna(input_signal)
    print("Input Signal:", input_signal)
    print("Output Signal after passing through Antenna:", output_signal)
