from abc import abstractmethod, ABC


class MABStrategy(ABC):
    """Abstract base class for MAB strategies."""
    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.reset()
    
    @abstractmethod
    def select_channel(self, time_step: int) -> int:
        """Select which channel to use."""
        pass
    
    @abstractmethod
    def update(self, channel: int, reward: float, time_step: int):
        """Update strategy based on observed reward."""
        pass
    
    def reset(self):
        """Reset strategy state."""
        pass
