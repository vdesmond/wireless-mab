from mab_strategy_abc import MABStrategy


class ConstantStrategy(MABStrategy):
    """Strategy that always selects a specific channel (no switching)"""
    def __init__(self, n_channels: int, constant_channel: int = 0):
        super().__init__(n_channels)
        self.constant_channel = constant_channel
    
    def select_channel(self, time_step: int) -> int:
        return self.constant_channel
    
    def update(self, channel: int, reward: float, time_step: int):
        pass
