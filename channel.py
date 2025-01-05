import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

class WirelessChannel:
    """
    Simulates a wireless channel with realistic characteristics including:
    - Time-varying SNR
    - Path loss
    - Shadow fading
    - Small-scale fading
    """
    def __init__(
        self,
        mean_snr_db: float = 20.0,
        shadow_std_db: float = 8.0,
        coherence_time: int = 50,
        bandwidth_mhz: float = 20.0
    ):
        self.mean_snr_db = mean_snr_db
        self.shadow_std_db = shadow_std_db
        self.coherence_time = coherence_time
        self.bandwidth_mhz = bandwidth_mhz
        
        self.current_shadow_fading = 0
        self.update_counter = 0
        
    def get_snr(self, time_step: int) -> float:
        """Calculate SNR at given time step including fading effects."""
        if time_step % self.coherence_time == 0:
            self.current_shadow_fading = np.random.normal(0, self.shadow_std_db)
        fast_fading_db = 10 * np.log10(np.random.rayleigh(scale=1.0))
        snr_db = self.mean_snr_db + self.current_shadow_fading + fast_fading_db
        return 10 ** (snr_db / 10)
    
    def get_throughput(self, time_step: int) -> float:
        """Calculate achievable throughput using Shannon capacity formula."""
        return self.bandwidth_mhz * np.log2(1 + self.get_snr(time_step))
    

def plot_channel_analysis(time_steps=500, window=10, n_runs=100):

    # NOTE: these are just sample values for demonstration purposes, and may be changed for later simulations to emphasise different aspects
    channels = [
        WirelessChannel(mean_snr_db=15.0),
        WirelessChannel(mean_snr_db=12.0),
        WirelessChannel(mean_snr_db=8.0)
    ]
    
    throughputs = np.zeros((3, time_steps))
    for t in range(time_steps):
        for i, ch in enumerate(channels):
            throughputs[i, t] = ch.get_throughput(t)
    
    # a moving average to smooth out the throughput values, as instantaneous values can be spiky (we really dont lose much information here)
    ma_throughputs = np.array([
        np.convolve(tput, np.ones(window)/window, mode='valid')
        for tput in throughputs
    ])
    ma_times = range(window-1, time_steps)
    optimal = np.max(ma_throughputs, axis=0)
    
    all_regrets = np.zeros((n_runs, 3, time_steps))
    for run in range(n_runs):
        run_throughputs = np.zeros((3, time_steps))
        for t in range(time_steps):
            for i, ch in enumerate(channels):
                run_throughputs[i, t] = ch.get_throughput(t)
        optimal_run = np.max(run_throughputs, axis=0)
        for i in range(3):
            all_regrets[run, i] = np.cumsum(optimal_run - run_throughputs[i])
    
    mean_regret = np.mean(all_regrets, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    labels = ['High SNR Channel', 'Medium SNR Channel', 'Low SNR Channel']

    for i in range(3):
        ax1.plot(ma_times, ma_throughputs[i], label=labels[i])
    ax1.plot(ma_times, optimal, '--w', label='Perfect Strategy', alpha=0.8)
    ax1.set_title('Channel Throughput (Moving Average)')
    ax1.set_ylabel('Throughput (Mbps)')
    ax1.legend()
    
    for i in range(3):
        ax2.plot(range(time_steps), mean_regret[i], 
                label=f'Always using {labels[i]}')
    ax2.plot(range(time_steps), np.zeros(time_steps), '--w', 
            label='Perfect Strategy', alpha=0.8)
    ax2.set_title('Mean Cumulative Regret (over {} runs)'.format(n_runs))
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative Regret (Mbps)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    font_manager.fontManager.addfont('Sora-variable.ttf')
    plt.rcParams['font.family'] = 'Sora'
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use("./style.mplstyle")
    
    fig = plot_channel_analysis()
    plt.show()
