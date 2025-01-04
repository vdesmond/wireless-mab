import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class WirelessMABEnvironment:
    """Multi-armed bandit environment for wireless channel selection."""
    def __init__(
        self,
        n_channels: int = 3,
        time_horizon: int = 1000,
        base_snr_range: Tuple[float, float] = (15.0, 25.0)
    ):
        self.n_channels = n_channels
        self.time_horizon = time_horizon
        
        # Initialize channels
        snrs = np.linspace(base_snr_range[0], base_snr_range[1], n_channels)
        self.channels = [
            WirelessChannel(mean_snr_db=snr) for snr in snrs
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
            
        # Get reward for selected channel
        reward = self.channels[action].get_throughput(self.current_time)
        
        # Track optimal reward
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

    def plot_results(self, results_dict: Dict, title: str = "Simulation Results"):
        """Plot simulation results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        steps = range(len(results_dict['rewards']))
        ax1.plot(steps, results_dict['rewards'], label='Achieved Throughput')
        ax1.plot(steps, results_dict['optimal_rewards'], 
                label='Optimal Throughput', linestyle='--')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Throughput (Mbps)')
        ax1.set_title(f'{title} - Throughput over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot cumulative regret
        ax2.plot(steps, results_dict['cumulative_regret'])
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Cumulative Regret')
        ax2.set_title('Cumulative Regret over Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
