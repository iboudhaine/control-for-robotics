"""
Excitation Signals for System Identification

Provides persistently exciting input signals for collecting informative data
from robotic systems. Persistent excitation is crucial for identifiability
in system identification.

Signals:
- Random (uniform/Gaussian)
- Sinusoidal (multi-frequency)
- Chirp (frequency sweep)
- PRBS (Pseudo-Random Binary Sequence)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class ExcitationSignal(ABC):
    """Base class for excitation signals."""
    
    def __init__(self, input_dim: int, seed: Optional[int] = None):
        """
        Args:
            input_dim: Dimension of the control input
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.rng = np.random.default_rng(seed)
        
    @abstractmethod
    def generate(self, t: float) -> np.ndarray:
        """Generate excitation signal at time t.
        
        Args:
            t: Current time
            
        Returns:
            Control input of shape (input_dim,)
        """
        pass
    
    def generate_sequence(self, times: np.ndarray) -> np.ndarray:
        """Generate signal sequence over time array.
        
        Args:
            times: Array of time points
            
        Returns:
            Control inputs of shape (len(times), input_dim)
        """
        return np.array([self.generate(t) for t in times])


class RandomExcitation(ExcitationSignal):
    """Random excitation signal (uniform or Gaussian).
    
    Provides basic persistent excitation through random inputs.
    Good for general system identification but may have high-frequency content.
    """
    
    def __init__(
        self,
        input_dim: int,
        amplitude: np.ndarray,
        distribution: str = 'uniform',
        hold_time: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            input_dim: Dimension of the control input
            amplitude: Max amplitude for each input channel, shape (input_dim,)
            distribution: 'uniform' or 'gaussian'
            hold_time: Duration to hold each random value (zero-order hold)
            seed: Random seed
        """
        super().__init__(input_dim, seed)
        self.amplitude = np.atleast_1d(amplitude)
        self.distribution = distribution
        self.hold_time = hold_time
        self._current_value = None
        self._last_switch_time = -np.inf
        
    def generate(self, t: float) -> np.ndarray:
        """Generate random signal with zero-order hold."""
        if t - self._last_switch_time >= self.hold_time:
            if self.distribution == 'uniform':
                self._current_value = self.rng.uniform(
                    -self.amplitude, self.amplitude
                )
            else:  # gaussian
                self._current_value = self.rng.normal(
                    scale=self.amplitude * 0.5
                )
                self._current_value = np.clip(
                    self._current_value, -self.amplitude, self.amplitude
                )
            self._last_switch_time = t
            
        return self._current_value.copy()


class SinusoidalExcitation(ExcitationSignal):
    """Multi-frequency sinusoidal excitation.
    
    Sum of sinusoids at different frequencies provides rich spectral content
    while maintaining smoothness. Each input channel can have different
    frequency content for better identifiability.
    """
    
    def __init__(
        self,
        input_dim: int,
        amplitudes: np.ndarray,
        frequencies: np.ndarray,
        phases: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            input_dim: Dimension of the control input
            amplitudes: Shape (input_dim, n_frequencies) - amplitude per channel per frequency
            frequencies: Shape (n_frequencies,) - frequencies in Hz
            phases: Shape (input_dim, n_frequencies) - phase offsets, random if None
            seed: Random seed
        """
        super().__init__(input_dim, seed)
        self.amplitudes = np.atleast_2d(amplitudes)
        self.frequencies = np.atleast_1d(frequencies)
        
        if phases is None:
            phases = self.rng.uniform(0, 2*np.pi, self.amplitudes.shape)
        self.phases = phases
        
    def generate(self, t: float) -> np.ndarray:
        """Generate multi-sine signal."""
        u = np.zeros(self.input_dim)
        for i in range(self.input_dim):
            for j, freq in enumerate(self.frequencies):
                u[i] += self.amplitudes[i, j] * np.sin(
                    2 * np.pi * freq * t + self.phases[i, j]
                )
        return u


class ChirpExcitation(ExcitationSignal):
    """Chirp (frequency sweep) excitation signal.
    
    Linear frequency sweep provides continuous spectral coverage.
    Excellent for identifying system frequency response.
    """
    
    def __init__(
        self,
        input_dim: int,
        amplitude: np.ndarray,
        f_start: float,
        f_end: float,
        duration: float,
        seed: Optional[int] = None
    ):
        """
        Args:
            input_dim: Dimension of control input
            amplitude: Amplitude per channel
            f_start: Starting frequency (Hz)
            f_end: Ending frequency (Hz)
            duration: Duration of one sweep
            seed: Random seed
        """
        super().__init__(input_dim, seed)
        self.amplitude = np.atleast_1d(amplitude)
        self.f_start = f_start
        self.f_end = f_end
        self.duration = duration
        # Random phase offset per channel
        self.phase_offset = self.rng.uniform(0, 2*np.pi, input_dim)
        
    def generate(self, t: float) -> np.ndarray:
        """Generate chirp signal."""
        # Wrap time for periodic behavior
        t_mod = t % self.duration
        
        # Linear chirp: frequency increases linearly with time
        k = (self.f_end - self.f_start) / self.duration
        phase = 2 * np.pi * (self.f_start * t_mod + 0.5 * k * t_mod**2)
        
        u = self.amplitude * np.sin(phase + self.phase_offset)
        return u


class PRBSExcitation(ExcitationSignal):
    """Pseudo-Random Binary Sequence (PRBS) excitation.
    
    Binary signal with well-defined spectral properties.
    Particularly useful for identification of linear systems
    and provides maximum information per unit energy.
    """
    
    def __init__(
        self,
        input_dim: int,
        amplitude: np.ndarray,
        switch_probability: float = 0.1,
        dt: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Args:
            input_dim: Dimension of control input
            amplitude: Amplitude (signal switches between +/- amplitude)
            switch_probability: Probability of switching per time step
            dt: Time step for switch decisions
            seed: Random seed
        """
        super().__init__(input_dim, seed)
        self.amplitude = np.atleast_1d(amplitude)
        self.switch_probability = switch_probability
        self.dt = dt
        self._current_value = self.amplitude.copy()
        self._last_time = 0.0
        
    def generate(self, t: float) -> np.ndarray:
        """Generate PRBS signal."""
        # Check if we should potentially switch
        n_steps = int((t - self._last_time) / self.dt)
        
        for _ in range(n_steps):
            for i in range(self.input_dim):
                if self.rng.random() < self.switch_probability:
                    self._current_value[i] *= -1
                    
        self._last_time = t
        return self._current_value.copy()


class CombinedExcitation(ExcitationSignal):
    """Combine multiple excitation signals.
    
    Allows mixing different signal types for comprehensive excitation.
    """
    
    def __init__(
        self,
        signals: list,
        weights: Optional[np.ndarray] = None
    ):
        """
        Args:
            signals: List of ExcitationSignal objects
            weights: Weights for combining signals (default: equal)
        """
        if len(signals) == 0:
            raise ValueError("At least one signal required")
            
        super().__init__(signals[0].input_dim, seed=None)
        self.signals = signals
        
        if weights is None:
            weights = np.ones(len(signals)) / len(signals)
        self.weights = weights
        
    def generate(self, t: float) -> np.ndarray:
        """Generate combined signal."""
        u = np.zeros(self.input_dim)
        for signal, weight in zip(self.signals, self.weights):
            u += weight * signal.generate(t)
        return u


def create_rich_excitation(
    input_dim: int,
    amplitude: np.ndarray,
    duration: float = 10.0,
    seed: Optional[int] = None
) -> CombinedExcitation:
    """Create a rich excitation signal combining multiple methods.
    
    Combines sinusoidal (low frequency), chirp (mid frequency), and
    random (broadband) for comprehensive persistent excitation.
    
    Args:
        input_dim: Dimension of control input
        amplitude: Max amplitude per channel
        duration: Duration for chirp sweep
        seed: Random seed
        
    Returns:
        CombinedExcitation signal
    """
    amplitude = np.atleast_1d(amplitude)
    rng = np.random.default_rng(seed)
    
    # Sinusoidal: low frequencies
    sin_amp = np.tile(amplitude[:, None] * 0.3, (1, 3))
    sin_freqs = np.array([0.1, 0.3, 0.5])
    sinusoidal = SinusoidalExcitation(
        input_dim, sin_amp, sin_freqs, 
        seed=rng.integers(10000)
    )
    
    # Chirp: mid frequencies
    chirp = ChirpExcitation(
        input_dim, amplitude * 0.3, 
        f_start=0.1, f_end=2.0, duration=duration,
        seed=rng.integers(10000)
    )
    
    # Random: broadband
    random = RandomExcitation(
        input_dim, amplitude * 0.4,
        distribution='uniform', hold_time=0.2,
        seed=rng.integers(10000)
    )
    
    return CombinedExcitation(
        [sinusoidal, chirp, random],
        weights=np.array([0.4, 0.3, 0.3])
    )
