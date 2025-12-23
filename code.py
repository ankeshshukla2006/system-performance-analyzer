import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
  

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics_df = None
        self.analysis = {}
    
    def generate_data(self, hours=24, interval=10):
        times = [datetime.now() - timedelta(minutes=i*interval) 
                for i in range(int(hours*60/interval))][::-1]
        
        data = {}
        for i, t in enumerate(times):
            # CPU with daily pattern
            hour_factor = 1.8 if 9 <= t.hour <= 17 else (1.3 if 18 <= t.hour <= 22 else 0.5)
            cpu = max(5, min(100, 30 + np.sin(i/50)*15 + np.random.randn()*5 + hour_factor*15))
            
            # Memory with slow leak
            mem = max(20, min(95, 45 + (i/len(times))*10 + np.random.randn()*3))
            
            # Disk I/O with periodic spikes
            disk = max(1, min(100, 15 * (1.5 if 9 <= t.hour <= 17 else 1) + 
                     (30 if t.hour % 2 == 0 and t.minute < 10 else 0) + np.random.randn()*2))
            
            # Network with occasional issues
            net = max(10, min(200, 25 + np.random.randn()*5 + (50 if np.random.rand() < 0.02 else 0)))
            
            data[t] = [cpu, mem, disk, net]
        
        self.metrics_df = pd.DataFrame(data.values(), index=data.keys(), 
                                      columns=['CPU', 'Memory', 'Disk', 'Network'])
        return self.metrics_df
    
    def analyze(self):
        for col in self.metrics_df.columns:
            d = self.metrics_df[col]
            self.analysis[col] = {
                'mean': np.mean(d), 'max': np.max(d), 'min': np.min(d),
                'std': np.std(d), 'p95': np.percentile(d, 95),
                'high': len(d[d > (80 if 'Network' not in col else 100)])
            }
        
        print("="*50 + "\nPERFORMANCE ANALYSIS\n" + "="*50)
        for m, s in self.analysis.items():
            print(f"{m:8s} | Avg: {s['mean']:5.1f} | Max: {s['max']:5.1f} | "
                  f"Std: {s['std']:4.1f} | >Thresh: {s['high']:3d}")
        return self.analysis
    
    def plot_time_series(self):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (ax, col, color) in enumerate(zip(axes, self.metrics_df.columns, colors)):
            ax.plot(self.metrics_df.index, self.metrics_df[col], color=color, linewidth=1.5)
            
            if col in ['CPU', 'Memory']:
                ax.axhline(80, color='orange', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(90, color='red', linestyle='--', alpha=0.5, linewidth=1)
            elif col == 'Network':
                ax.axhline(100, color='orange', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(150, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.fill_between(self.metrics_df.index, 0, self.metrics_df[col], 
                           alpha=0.1, color=color)
            ax.set_ylabel(col, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if i == 3:
                ax.set_xlabel('Time')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Performance Metrics Over Time', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_distributions(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for ax, col, color in zip(axes.flatten(), self.metrics_df.columns, colors):
            data = self.metrics_df[col]
            ax.hist(data, bins=25, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(np.mean(data), color='red', linewidth=2, label=f'Mean: {np.mean(data):.1f}')
            
            if col in ['CPU', 'Memory']:
                ax.axvline(80, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
                ax.axvline(90, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
            
            ax.set_title(f'{col} Distribution', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Distributions', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_summary(self):
        fig = plt.figure(figsize=(14, 10))
        
        # Time series
        ax1 = plt.subplot(2, 2, 1)
        for col in self.metrics_df.columns[:2]:
            ax1.plot(self.metrics_df.index, self.metrics_df[col], label=col, linewidth=1.5)
        ax1.set_title('CPU & Memory Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Distributions
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(self.metrics_df['CPU'], bins=25, alpha=0.7, label='CPU')
        ax2.hist(self.metrics_df['Memory'], bins=25, alpha=0.7, label='Memory')
        ax2.set_title('CPU vs Memory Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Box plot
        ax3 = plt.subplot(2, 2, 3)
        bp = ax3.boxplot([self.metrics_df[col] for col in self.metrics_df.columns], 
                         labels=self.metrics_df.columns, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_title('Metrics Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Correlation
        ax4 = plt.subplot(2, 2, 4)
        corr = self.metrics_df.corr()
        im = ax4.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax4)
        ax4.set_xticks(range(len(corr.columns)))
        ax4.set_yticks(range(len(corr.columns)))
        ax4.set_xticklabels(corr.columns, rotation=45)
        ax4.set_yticklabels(corr.columns)
        ax4.set_title('Correlation Matrix', fontweight='bold')
        
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax4.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', 
                        color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')
        
        plt.suptitle('Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def main():
    analyzer = PerformanceAnalyzer()
    analyzer.generate_data(hours=12, interval=5)
    analyzer.analyze()
    analyzer.plot_time_series()
    analyzer.plot_distributions()
    analyzer.plot_summary()

if __name__ == "__main__":
    main()
