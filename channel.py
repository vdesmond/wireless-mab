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
        shadow_std_db: float = 4.0,
        coherence_time: int = 100,
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
        snr = self.get_snr(time_step)
        throughput = self.bandwidth_mhz * np.log2(1 + snr)
        return throughput
    

def plot_channels(time_steps=50):
    channels = [
        WirelessChannel(mean_snr_db=20.0, shadow_std_db=8.0, coherence_time=10),
        WirelessChannel(mean_snr_db=15.0, shadow_std_db=8.0, coherence_time=10),
        WirelessChannel(mean_snr_db=10.0, shadow_std_db=8.0, coherence_time=10)
    ]

    times = range(time_steps)
    labels = ['High SNR Channel', 'Medium SNR Channel', 'Low SNR Channel']
    
    fig, (ax1, ax2) = plt.subplots(2, 1)

    best_throughput = np.zeros(time_steps)
    high_snr_throughput = np.zeros(time_steps)

    for t in times:
        throughputs = [ch.get_throughput(t) for ch in channels]
        best_throughput[t] = max(throughputs)
        high_snr_throughput[t] = throughputs[0]

    for i, channel in enumerate(channels):
        throughput = [channel.get_throughput(t) for t in times]
        ax1.plot(throughput, label=labels[i])

    cumulative_regret = np.cumsum(best_throughput - high_snr_throughput)
    ax2.plot(cumulative_regret, label='Regret from always choosing high SNR')

    ax1.set_title('Channel Throughput')
    ax1.set_ylabel('Throughput (Mbps)')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative Regret')

    ax1.legend()
    ax2.legend()

    plt.tight_layout(pad=2.0)
    return fig

if __name__ == "__main__":
    font_manager.fontManager.addfont('Sora-variable.ttf')
    plt.rcParams['font.family'] = 'Sora'
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use("./style.mplstyle")
    fig = plot_channels()
    plt.show()
