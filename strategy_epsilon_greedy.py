from matplotlib import font_manager, pyplot as plt
import numpy as np
from mab_environment import WirelessMABEnvironment
from mab_strategy_abc import MABStrategy
from strategy_constant import ConstantStrategy


class EpsilonGreedyCumulative(MABStrategy):
    def __init__(self, n_channels: int, epsilon: float = 0.1):
        super().__init__(n_channels)
        self.epsilon = epsilon
        
    def reset(self):
        self.channel_rewards = np.zeros(self.n_channels)
        self.channel_counts = np.zeros(self.n_channels)
    
    def select_channel(self, time_step: int) -> int:
        untried = np.where(self.channel_counts == 0)[0]
        if len(untried) > 0:
            return np.random.choice(untried)
            
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_channels)
            
        averages = self.channel_rewards / self.channel_counts
        best_channels = np.where(averages == averages.max())[0]
        return np.random.choice(best_channels)
    
    def update(self, channel: int, reward: float, time_step: int):
        self.channel_rewards[channel] += reward
        self.channel_counts[channel] += 1

class EpsilonGreedyRecency(MABStrategy):
    def __init__(self, n_channels: int, epsilon: float = 0.1, alpha: float = 0.1):
        super().__init__(n_channels)
        self.epsilon = epsilon
        self.alpha = alpha
        
    def reset(self):
        self.q_values = np.zeros(self.n_channels)
        self.channel_counts = np.zeros(self.n_channels)
    
    def select_channel(self, time_step: int) -> int:
        untried = np.where(self.channel_counts == 0)[0]
        if len(untried) > 0:
            return np.random.choice(untried)
            
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_channels)
            
        best_channels = np.where(self.q_values == self.q_values.max())[0]
        return np.random.choice(best_channels)
    
    def update(self, channel: int, reward: float, time_step: int):
        self.q_values[channel] = ((1 - self.alpha) * self.q_values[channel] + 
                                 self.alpha * reward)
        self.channel_counts[channel] += 1


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
            
            marker = 'D' if isinstance(strategy, EpsilonGreedyCumulative) else 's'
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

def plot_alpha_comparisons(time_steps=500, n_runs=100):
    env = WirelessMABEnvironment(n_channels=3, time_horizon=time_steps, snrs=(15.0, 12.0, 8.0), coherence_time=100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = [
        (ConstantStrategy(3, 0), r"Always High SNR"),
        (EpsilonGreedyRecency(3, 0.1, 0.1), r"$\alpha=0.1$"),
        (EpsilonGreedyRecency(3, 0.1, 0.2), r"$\alpha=0.2$"),
        (EpsilonGreedyRecency(3, 0.1, 0.3), r"$\alpha=0.3$"),
        (EpsilonGreedyRecency(3, 0.1, 0.5), r"$\alpha=0.5$"),
        (EpsilonGreedyRecency(3, 0.1, 1), r"$\alpha=1.0$"),
    ]
    
    plot_strategies(env, strategies, ax, n_runs)
    ax.set_title(r"Cumulative regret - different $\alpha$ Values")
    ax.set_xlabel(r"Time Step ($t$)")
    ax.set_ylabel(r"Mean Cumulative Regret")
    plt.tight_layout()
    return fig

def plot_strategy_comparisons(time_steps=500, n_runs=100):
    env = WirelessMABEnvironment(n_channels=3, time_horizon=time_steps, snrs=(15.0, 12.0, 8.0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = [
        (ConstantStrategy(3, 0), r"Always High SNR"),
        (EpsilonGreedyCumulative(3, 0.1), r"$\varepsilon=0.1$ (cumulative)"),
        (EpsilonGreedyCumulative(3, 0.3), r"$\varepsilon=0.3$ (cumulative)"),
        (EpsilonGreedyCumulative(3, 0.5), r"$\varepsilon=0.5$ (cumulative)"),
        (EpsilonGreedyCumulative(3, 1.0), r"$\varepsilon=1.0$ (cumulative)"),
        (EpsilonGreedyRecency(3, 0.1, 0.5), r"$\varepsilon=0.1$ (ES)"),
        (EpsilonGreedyRecency(3, 0.3, 0.5), r"$\varepsilon=0.3$ (ES)"),
        (EpsilonGreedyRecency(3, 0.5, 0.5), r"$\varepsilon=0.5$ (ES)"),
        (EpsilonGreedyRecency(3, 1.0, 0.5), r"$\varepsilon=1.0$ (ES)")
    ]
    
    plot_strategies(env, strategies, ax, n_runs)
    ax.set_title(r"Cumulative regret - $\varepsilon$-greedy with different $\varepsilon$ values")
    ax.set_xlabel(r"Time Step ($t$)")
    ax.set_ylabel(r"Mean Cumulative Regret")
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    font_manager.fontManager.addfont('Sora-variable.ttf')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Sora', 'Inter']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use("./style.mplstyle")
    fig1 = plot_alpha_comparisons()
    fig2 = plot_strategy_comparisons()
    plt.show()

