#!/usr/bin/env python3
"""
Automated Resonance Detection System
A cymatic pattern hunter that scans frequency space for emergent phenomena
"""

import numpy as np
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class PatternSignature:
    """Represents a detected resonance pattern"""
    frequencies: List[float]
    harmonic_ratio: float
    complexity_score: float
    stability_score: float
    beauty_score: float
    timestamp: float
    duration: float
    pattern_type: str
    
    def __str__(self):
        return f"{self.pattern_type}: {self.frequencies} (beauty={self.beauty_score:.3f})"

class ResonanceDetector:
    """Core pattern detection engine for resonance phenomena"""
    
    def __init__(self, sample_rate=1000, window_size=512):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.history_length = 100
        
        # Pattern detection parameters
        self.beauty_threshold = 0.7
        self.stability_threshold = 0.8
        self.min_pattern_duration = 2.0
        
        # Internal state
        self.signal_history = deque(maxlen=self.history_length)
        self.detected_patterns = []
        self.current_pattern_start = None
        self.last_analysis_time = 0
        
        # Golden ratio and other special numbers
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.special_ratios = [1, 2, 3/2, 4/3, 5/4, self.phi, 2/self.phi, math.sqrt(2), math.pi/2]
    
    def analyze_signal(self, signal: np.ndarray, current_frequencies: List[float]) -> Optional[PatternSignature]:
        """Analyze current signal for interesting patterns"""
        
        # Store signal history
        self.signal_history.append(signal.copy())
        
        if len(self.signal_history) < 10:  # Need some history
            return None
            
        # Calculate various pattern metrics
        harmonic_score = self.calculate_harmonic_score(current_frequencies)
        complexity_score = self.calculate_complexity_score(signal)
        stability_score = self.calculate_stability_score()
        coherence_score = self.calculate_coherence_score(signal)
        symmetry_score = self.calculate_symmetry_score(signal)
        
        # Combined beauty score
        beauty_score = self.calculate_beauty_score(
            harmonic_score, complexity_score, stability_score, 
            coherence_score, symmetry_score
        )
        
        # Check if this is an interesting pattern
        if beauty_score > self.beauty_threshold and stability_score > self.stability_threshold:
            pattern_type = self.classify_pattern_type(harmonic_score, complexity_score, current_frequencies)
            
            current_time = time.time()
            
            if self.current_pattern_start is None:
                self.current_pattern_start = current_time
                
            pattern_duration = current_time - self.current_pattern_start
            
            if pattern_duration > self.min_pattern_duration:
                pattern = PatternSignature(
                    frequencies=current_frequencies.copy(),
                    harmonic_ratio=harmonic_score,
                    complexity_score=complexity_score,
                    stability_score=stability_score,
                    beauty_score=beauty_score,
                    timestamp=self.current_pattern_start,
                    duration=pattern_duration,
                    pattern_type=pattern_type
                )
                
                return pattern
        else:
            self.current_pattern_start = None
            
        return None
    
    def calculate_harmonic_score(self, frequencies: List[float]) -> float:
        """Calculate how harmonically related the frequencies are"""
        if len(frequencies) < 2:
            return 0.0
            
        scores = []
        
        # Check all pairs of frequencies
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                f1, f2 = frequencies[i], frequencies[j]
                if f1 > 0 and f2 > 0:
                    ratio = max(f1, f2) / min(f1, f2)
                    
                    # Check against special ratios
                    best_score = 0
                    for special_ratio in self.special_ratios:
                        error = abs(ratio - special_ratio) / special_ratio
                        if error < 0.05:  # Within 5%
                            score = 1.0 - error * 20  # Scale error to score
                            best_score = max(best_score, score)
                    
                    scores.append(best_score)
        
        return np.mean(scores) if scores else 0.0
    
    def calculate_complexity_score(self, signal: np.ndarray) -> float:
        """Calculate complexity - not too simple, not too chaotic"""
        
        # FFT analysis
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft[:len(fft)//2])
        
        # Normalize
        if np.max(power_spectrum) > 0:
            power_spectrum = power_spectrum / np.max(power_spectrum)
        
        # Calculate spectral entropy
        eps = 1e-12
        power_spectrum = power_spectrum + eps
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        entropy = -np.sum(power_spectrum * np.log2(power_spectrum))
        
        # Normalize entropy (max possible for this length)
        max_entropy = math.log2(len(power_spectrum))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Optimal complexity is around 0.6-0.8 (neither too simple nor too random)
        if normalized_entropy < 0.6:
            return normalized_entropy / 0.6
        elif normalized_entropy > 0.8:
            return (1.0 - normalized_entropy) / 0.2
        else:
            return 1.0  # In the sweet spot
    
    def calculate_stability_score(self) -> float:
        """Calculate how stable the pattern is over time"""
        if len(self.signal_history) < 5:
            return 0.0
            
        # Compare recent signals for similarity
        recent_signals = list(self.signal_history)[-5:]
        correlations = []
        
        for i in range(len(recent_signals) - 1):
            corr = np.corrcoef(recent_signals[i], recent_signals[i + 1])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_coherence_score(self, signal: np.ndarray) -> float:
        """Calculate internal coherence of the signal"""
        # Autocorrelation to find repeating patterns
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find the strongest non-zero lag correlation
        if len(autocorr) > 1:
            # Normalize
            autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
            # Find peak (excluding lag 0)
            peak_idx = np.argmax(autocorr[1:]) + 1
            return autocorr[peak_idx] if peak_idx < len(autocorr) else 0.0
        return 0.0
    
    def calculate_symmetry_score(self, signal: np.ndarray) -> float:
        """Calculate symmetry in the signal"""
        # Check for reflection symmetry
        reversed_signal = signal[::-1]
        correlation = np.corrcoef(signal, reversed_signal)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_beauty_score(self, harmonic: float, complexity: float, 
                              stability: float, coherence: float, symmetry: float) -> float:
        """Combine all metrics into a single beauty score"""
        
        # Weighted combination emphasizing harmony and stability
        weights = {
            'harmonic': 0.3,
            'complexity': 0.2,
            'stability': 0.25,
            'coherence': 0.15,
            'symmetry': 0.1
        }
        
        beauty = (weights['harmonic'] * harmonic +
                 weights['complexity'] * complexity +
                 weights['stability'] * stability +
                 weights['coherence'] * coherence +
                 weights['symmetry'] * symmetry)
        
        return min(1.0, max(0.0, beauty))
    
    def classify_pattern_type(self, harmonic_score: float, complexity_score: float, 
                            frequencies: List[float]) -> str:
        """Classify the type of pattern detected"""
        
        if harmonic_score > 0.9:
            if len(frequencies) >= 2:
                ratio = max(frequencies) / min(frequencies) if min(frequencies) > 0 else 0
                if abs(ratio - 2.0) < 0.1:
                    return "OCTAVE_HARMONY"
                elif abs(ratio - 1.5) < 0.1:
                    return "PERFECT_FIFTH"
                elif abs(ratio - self.phi) < 0.1:
                    return "GOLDEN_RESONANCE"
                else:
                    return "HARMONIC_SERIES"
            return "PURE_HARMONY"
        elif complexity_score > 0.8 and harmonic_score > 0.5:
            return "COMPLEX_HARMONY"
        elif complexity_score > 0.7:
            return "CHAOTIC_BEAUTY"
        elif harmonic_score > 0.6:
            return "SIMPLE_RESONANCE"
        else:
            return "EMERGENT_PATTERN"

class FrequencyScanner:
    """Automated frequency space explorer"""
    
    def __init__(self, detector: ResonanceDetector):
        self.detector = detector
        self.scan_range = (0.1, 10.0)  # Hz
        self.scan_resolution = 0.05
        self.scan_speed = 0.1  # Hz per second
        self.dwell_time = 3.0  # Seconds to analyze each frequency
        
        # Current scan state
        self.current_freq_1 = 1.0
        self.current_freq_2 = 1.618
        self.current_freq_3 = 0.618
        self.scan_direction = 1
        self.last_scan_time = time.time()
        self.locked_pattern = None
        self.lock_start_time = None
        
    def generate_signal(self, duration: float = 1.0) -> Tuple[np.ndarray, List[float]]:
        """Generate current signal based on scanner frequencies"""
        
        sample_rate = self.detector.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate multi-frequency signal
        signal = (np.sin(2 * np.pi * self.current_freq_1 * t) +
                 0.7 * np.sin(2 * np.pi * self.current_freq_2 * t) +
                 0.5 * np.sin(2 * np.pi * self.current_freq_3 * t))
        
        frequencies = [self.current_freq_1, self.current_freq_2, self.current_freq_3]
        return signal, frequencies
    
    def update_scan(self) -> Optional[PatternSignature]:
        """Update scanner state and check for patterns"""
        
        current_time = time.time()
        dt = current_time - self.last_scan_time
        
        # Generate and analyze current signal
        signal, frequencies = self.generate_signal(0.5)
        pattern = self.detector.analyze_signal(signal, frequencies)
        
        # If we found an interesting pattern, lock onto it
        if pattern and not self.locked_pattern:
            self.locked_pattern = pattern
            self.lock_start_time = current_time
            print(f">>> LOCK: {pattern}")
            return pattern
        
        # If locked, stay locked for a while to fully analyze
        if self.locked_pattern:
            lock_duration = current_time - self.lock_start_time
            if lock_duration > 10.0:  # Lock for 10 seconds
                print(f">>> Logged: {self.locked_pattern}")
                self.detector.detected_patterns.append(self.locked_pattern)
                self.locked_pattern = None
                return None
            else:
                return None  # Stay locked
        
        # Continue scanning if not locked
        if dt > 0.1:  # Update every 100ms
            self.advance_scan(dt)
            self.last_scan_time = current_time
            
        return None
    
    def advance_scan(self, dt: float):
        """Advance the frequency scan"""
        
        # Scan the second frequency while keeping first and third in golden ratio
        freq_change = self.scan_speed * dt * self.scan_direction
        self.current_freq_2 += freq_change
        
        # Reverse direction at boundaries
        if self.current_freq_2 >= self.scan_range[1]:
            self.current_freq_2 = self.scan_range[1]
            self.scan_direction = -1
        elif self.current_freq_2 <= self.scan_range[0]:
            self.current_freq_2 = self.scan_range[0]
            self.scan_direction = 1
        
        # Keep other frequencies in interesting relationships
        self.current_freq_1 = max(0.1, self.current_freq_2 / self.detector.phi)
        self.current_freq_3 = self.current_freq_2 / 2.0

def main():
    """Test the resonance detection system"""
    detector = ResonanceDetector()
    scanner = FrequencyScanner(detector)
    
    print(">>> Automated Resonance Detection System")
    print("Scanning frequency space for emergent patterns...")
    print("=" * 50)
    
    try:
        for i in range(100):  # Run for a short test
            pattern = scanner.update_scan()
            
            # Show current scan state
            if i % 50 == 0:
                freqs = [scanner.current_freq_1, scanner.current_freq_2, scanner.current_freq_3]
                print(f"Scanning: {[f'{f:.2f}' for f in freqs]} Hz")
            
            time.sleep(0.05)  # 20 FPS
            
    except KeyboardInterrupt:
        print(f"\n>>> Scan complete. Found {len(detector.detected_patterns)} patterns:")
        for i, pattern in enumerate(detector.detected_patterns):
            print(f"{i+1}: {pattern}")

if __name__ == "__main__":
    main()