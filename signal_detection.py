import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        for an_input in [hits, misses, false_alarms, correct_rejections]:
            if not (type(an_input) == int or type(an_input) == float):
                raise ValueError("All inputs must be numerical.")
        for an_input in [hits, misses, false_alarms, correct_rejections]:
            if an_input < 0:
                raise ValueError("Inputs cannot be negative.")
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections
    
    def hit_rate(self):
        denom = self.hits + self.misses
        if denom == 0:
            raise ValueError("Cannot calculate hit rate: hits + misses = 0.")
        return self.hits / denom
    
    def false_alarm_rate(self):
        denom = self.false_alarms + self.correct_rejections
        if denom == 0:
            raise ValueError("Cannot calculate false alarm rate: false_alarms + correct_rejections = 0.")
        return self.false_alarms / denom
    
    def d_prime(self):
        h = self.hit_rate()
        fa = self.false_alarm_rate()
        
        if h == 0 or h == 1:
            raise ValueError("Cannot calculate d': hit rate is exactly 0 or 1.")
        if fa == 0 or fa == 1:
            raise ValueError("Cannot calculate d': false alarm rate is exactly 0 or 1.")
        
        return norm.ppf(h) - norm.ppf(fa)
    
    def criterion(self):
        h = self.hit_rate()
        fa = self.false_alarm_rate()
        
        if h == 0 or h == 1:
            raise ValueError("Cannot calculate C: hit rate is exactly 0 or 1.")
        if fa == 0 or fa == 1:
            raise ValueError("Cannot calculate C: false alarm rate is exactly 0 or 1.")
        
        return -0.5 * (norm.ppf(h) + norm.ppf(fa))
    
    def __add__(self, other):
        if type(other) != SignalDetection:
            raise TypeError("Can only add 'SignalDetection'-type objects.")
        
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_rejections + other.correct_rejections)
    
    def __sub__(self, other):
        if type(other) != SignalDetection:
            raise TypeError("Can only subtract 'SignalDetection'-type objects.")
        
        return SignalDetection(
            self.hits - other.hits,
            self.misses - other.misses,
            self.false_alarms - other.false_alarms,
            self.correct_rejections - other.correct_rejections)
    
    def __mul__(self, factor):
        if not (type(factor) == int or type(factor) == float):
            raise TypeError("Can only multiply using a numerical scalar.")
        
        return SignalDetection(
            self.hits * factor,
            self.misses * factor,
            self.false_alarms * factor,
            self.correct_rejections * factor)
    
    def plot_sdt(self):
        dprime = self.d_prime()
        c = self.criterion()
        
        x = np.linspace(-4, 4 + dprime, 1000)
        noise = norm.pdf(x, 0, 1)
        signal = norm.pdf(x, dprime, 1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, noise, label='Noise', color='red', linewidth=2)
        ax.plot(x, signal, label='Signal', color='blue', linewidth=2)
        ax.axvline(c, color='green', linestyle='--', linewidth=2, label='Criterion')
        ax.annotate('', xy=(dprime, 0.05), xytext=(0, 0.05), arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(dprime/2, 0.05 + 0.02, f"d'={dprime:.2f}", ha='center', fontsize=12)
        
        ax.set_xlabel('Internal response', fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)
        ax.set_title('SDT plot', fontsize=15)
        ax.legend(fontsize=10)
        
        return fig, ax
    
    @staticmethod
    def plot_roc(sdt_list):
        if not (type(sdt_list) == list or type(sdt_list) == tuple):
            raise TypeError("sdt_list must be either a list or a tuple.")
        for sd in sdt_list:
            if type(sd) != SignalDetection:
                raise TypeError("All elements of sdt_list must be 'SignalDetection'-type objects.")
    
        hit_rates = [0] + [sd.hit_rate() for sd in sdt_list] + [1]
        fa_rates = [0] + [sd.false_alarm_rate() for sd in sdt_list] + [1]
    
        pairs = []
        for i in range(len(hit_rates)):
            pairs.append((hit_rates[i], fa_rates[i]))
        pairs.sort()
    
        hit_rates_sorted = []
        fa_rates_sorted = []
        for pair in pairs:
            hit_rates_sorted.append(pair[0])
            fa_rates_sorted.append(pair[1])
    
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(hit_rates_sorted, fa_rates_sorted, 'ro-', linewidth=2, markersize=6)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
        ax.set_xlabel('Hit rate', fontsize=12)
        ax.set_ylabel('False alarm rate', fontsize=12)
        ax.set_title('ROC curve plot', fontsize=15, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
    
        return fig, ax