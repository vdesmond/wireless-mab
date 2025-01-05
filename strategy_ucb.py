import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from mab_environment import WirelessMABEnvironment
from mab_strategy_abc import MABStrategy
from strategy_constant import ConstantStrategy
from strategy_epsilon_greedy import EpsilonGreedyRecency

class UCBStrategy(MABStrategy):
    def __init__(self, n_channels: int, alpha: float = 0.5):
        super().__init__(n_channels)
        self.alpha = alpha
        
    def reset(self):
        self.q_values = np.zeros(self.n_channels)
        self.channel_counts = np.zeros(self.n_channels)
        
    def select_channel(self, time_step: int) -> int:
        untried = np.where(self.channel_counts == 0)[0]
        if len(untried) > 0:
            return np.random.choice(untried)
            
        scale = np.mean(self.q_values) or 1.0
        exploration_bonus = scale * np.sqrt(2 * np.log(time_step) / self.channel_counts)
        ucb_values = self.q_values + exploration_bonus
        
        best_channels = np.where(ucb_values == ucb_values.max())[0]
        return np.random.choice(best_channels)
        
    def update(self, channel: int, reward: float, time_step: int):
        self.q_values[channel] = ((1 - self.alpha) * self.q_values[channel] + 
                                 self.alpha * reward)
        self.channel_counts[channel] += 1

class ThompsonSamplingRecency(MABStrategy):
    def __init__(self, n_channels: int, alpha: float = 0.1):
        super().__init__(n_channels)
        self.alpha = alpha
        
    def reset(self):
        self.q_values = np.zeros(self.n_channels)
        self.uncertainties = np.ones(self.n_channels)
        
    def select_channel(self, time_step: int) -> int:
        samples = np.random.normal(self.q_values, self.uncertainties)
        return np.argmax(samples)
        
    def update(self, channel: int, reward: float, time_step: int):
        # Update mean estimate with exponential smoothing
        self.q_values[channel] = (1 - self.alpha) * self.q_values[channel] + self.alpha * reward
        
        # Update uncertainty estimate
        error = abs(reward - self.q_values[channel])
        self.uncertainties[channel] = (1 - self.alpha) * self.uncertainties[channel] + self.alpha * error

def plot_strategies(env, strategies, ax, n_runs):
    """Plot strategies comparison with given number of runs"""
    ax.grid(True, alpha=0.2)
    time_steps = env.time_horizon
    
    all_regrets = {name: np.zeros((n_runs, time_steps)) for _, name in strategies}
    
    for run in range(n_runs):
        for strategy, name in strategies:
            history = env.run_simulation(strategy)
            all_regrets[name][run] = history['cumulative_regret']
    
    for strategy, name in strategies:
        mean_regret = np.mean(all_regrets[name], axis=0)
        
        if isinstance(strategy, ConstantStrategy):
            ax.plot(range(time_steps), mean_regret, '--w', 
                    label=name, linewidth=1)
        else:
            std_error = np.std(all_regrets[name], axis=0) / np.sqrt(n_runs)
            ci_95 = 1.96 * std_error
            
            marker = 'D' if isinstance(strategy, EpsilonGreedyRecency) else 's'
            markevery = time_steps // 20
            
            line = ax.plot(range(time_steps), mean_regret, 
                            label=name, linewidth=1,
                            marker=marker, markevery=markevery)
            
            ax.fill_between(range(time_steps), 
                           mean_regret - ci_95,
                           mean_regret + ci_95,
                           alpha=0.2)
    
    ax.legend(loc='upper left')
    return ax

def plot_ucb_comparison(time_steps=500, n_runs=100):
    env = WirelessMABEnvironment(n_channels=3, time_horizon=time_steps, 
                                snrs=(15.0, 12.0, 8.0), coherence_time=100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = [
        (ConstantStrategy(3, 0), "Always High SNR"),
        (UCBStrategy(3, 0.5), r"UCB $\alpha$=0.5"),
        (UCBStrategy(3, 0.3), r"UCB $\alpha$=0.3"),
        (UCBStrategy(3, 0.2), r"UCB $\alpha$=0.2"),
        (UCBStrategy(3, 0.1), r"UCB $\alpha$=0.1"),
        (UCBStrategy(3, 1), r"UCB $\alpha$=1"),
        (EpsilonGreedyRecency(3, 0.1, 0.5), r"$\varepsilon$-greedy $\varepsilon$=0.1 $\alpha$=0.5"),
    ]
    
    plot_strategies(env, strategies, ax, n_runs)
    ax.set_title("Cumulative Regret - UCB Comparison (with different $\\alpha$ values)")
    ax.set_xlabel(r"Time Step ($t$)")
    ax.set_ylabel("Mean Cumulative Regret")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    font_manager.fontManager.addfont('Sora-variable.ttf')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Sora', 'Inter']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use("./style.mplstyle")
    
    fig = plot_ucb_comparison()
    plt.show()
