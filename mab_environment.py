import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt

from channel import WirelessChannel
from mab_strategy_abc import MABStrategy

class WirelessMABEnvironment:
    """Multi-armed bandit environment for wireless channel selection."""
    def __init__(
        self,
        n_channels: int = 3,
        time_horizon: int = 500,
        snrs: Tuple[float] = (15.0, 12.0, 10.0),
        shadow_std_db: float = 8.0,
        coherence_time: int = 100,
    ):
        self.n_channels = n_channels
        self.time_horizon = time_horizon

        if len(snrs) != n_channels:
            raise ValueError("Number of SNRs must match number of channels")

        self.channels = [
            WirelessChannel(mean_snr_db=snr, shadow_std_db=shadow_std_db, coherence_time=coherence_time) for snr in snrs
        ]
        
        self.reset()
    
    def reset(self):
        """Reset environment state."""
        self.current_time = 0
        self.history = {
            'rewards': [],
            'actions': [],
            'optimal_rewards': [],
            'cumulative_regret': []
        }
        
    def step(self, action: int) -> float:
        """Execute one step in the environment."""
        if self.current_time >= self.time_horizon:
            raise ValueError("Time horizon exceeded")
            
        reward = self.channels[action].get_throughput(self.current_time)
        
        optimal_reward = max(
            ch.get_throughput(self.current_time) for ch in self.channels
        )
        
        # Update history
        self.history['rewards'].append(reward)
        self.history['actions'].append(action)
        self.history['optimal_rewards'].append(optimal_reward)
        
        # Update cumulative regret
        prev_regret = (self.history['cumulative_regret'][-1] 
                      if self.history['cumulative_regret'] else 0)
        current_regret = optimal_reward - reward
        self.history['cumulative_regret'].append(prev_regret + current_regret)
        
        self.current_time += 1
        return reward

    def run_simulation(self, strategy: MABStrategy) -> Dict:
        """Run a complete simulation with given strategy."""
        self.reset()
        strategy.reset()
        
        while self.current_time < self.time_horizon:
            action = strategy.select_channel(self.current_time)
            reward = self.step(action)
            strategy.update(action, reward, self.current_time)
            
        return self.history
    
    def get_best_channel(self, time_step: int) -> int:
        """Returns index of best channel at current time step."""
        throughputs = [ch.get_throughput(time_step) for ch in self.channels]
        return np.argmax(throughputs)
