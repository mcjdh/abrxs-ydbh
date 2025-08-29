#!/usr/bin/env python3
"""
Living Oscilloscope - A Dynamic Resonance Explorer
Waves that evolve, flow, and lock onto mathematical beauty
"""

import tkinter as tk
import numpy as np
import math
import time
import threading
from queue import Queue
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class WaveGenome:
    """Genetic representation of a wave pattern"""
    frequencies: List[float]
    amplitudes: List[float] 
    phases: List[float]
    modulation_rates: List[float]
    fitness: float = 0.0
    age: int = 0
    
    def mutate(self, strength=0.1):
        """Evolve the wave parameters"""
        new_genome = WaveGenome(
            frequencies=self.frequencies.copy(),
            amplitudes=self.amplitudes.copy(),
            phases=self.phases.copy(),
            modulation_rates=self.modulation_rates.copy(),
            fitness=self.fitness,
            age=self.age + 1
        )
        
        # Mutate frequencies with bias toward harmonic ratios
        for i in range(len(new_genome.frequencies)):
            if random.random() < 0.3:  # 30% chance to mutate each frequency
                if random.random() < 0.5:
                    # Small random drift
                    new_genome.frequencies[i] *= (1 + random.gauss(0, strength))
                else:
                    # Jump to harmonic ratio
                    base_freq = new_genome.frequencies[0]
                    ratios = [0.5, 2/3, 0.75, 1.0, 1.25, 1.5, 1.618, 2.0, 2.5, 3.0]
                    new_genome.frequencies[i] = base_freq * random.choice(ratios)
        
        # Mutate amplitudes
        for i in range(len(new_genome.amplitudes)):
            if random.random() < 0.2:
                new_genome.amplitudes[i] *= (1 + random.gauss(0, strength * 0.5))
                new_genome.amplitudes[i] = max(0.1, min(2.0, new_genome.amplitudes[i]))
        
        # Mutate phases
        for i in range(len(new_genome.phases)):
            if random.random() < 0.2:
                new_genome.phases[i] += random.gauss(0, math.pi * strength)
        
        # Mutate modulation rates
        for i in range(len(new_genome.modulation_rates)):
            if random.random() < 0.1:
                new_genome.modulation_rates[i] += random.gauss(0, strength * 0.5)
        
        return new_genome

class GeneticWaveGenerator:
    """Evolves wave patterns toward mathematical beauty"""
    
    def __init__(self, population_size=20):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.current_champion = None
        self.lock_duration = 0
        self.exploration_pressure = 1.0
        
        # Initialize population
        self.spawn_initial_population()
    
    def spawn_initial_population(self):
        """Create initial wave population"""
        self.population = []
        for _ in range(self.population_size):
            genome = WaveGenome(
                frequencies=[random.uniform(0.5, 5.0) for _ in range(3)],
                amplitudes=[random.uniform(0.3, 1.0) for _ in range(3)],
                phases=[random.uniform(0, 2*math.pi) for _ in range(3)],
                modulation_rates=[random.uniform(-0.1, 0.1) for _ in range(3)]
            )
            self.population.append(genome)
    
    def evaluate_fitness(self, genome, signal):
        """Calculate fitness based on harmonic content and beauty"""
        
        # Harmonic analysis
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft[:len(fft)//2])
        
        if len(power_spectrum) == 0:
            return 0.0
        
        # Normalize
        power_spectrum = power_spectrum / (np.max(power_spectrum) + 1e-12)
        
        # Harmonic ratio fitness
        harmonic_fitness = 0.0
        golden_ratio = (1 + math.sqrt(5)) / 2
        special_ratios = [1.0, 2.0, 1.5, 4/3, 5/4, golden_ratio, 2/golden_ratio, math.sqrt(2)]
        
        for i in range(len(genome.frequencies)):
            for j in range(i + 1, len(genome.frequencies)):
                if genome.frequencies[j] > 0 and genome.frequencies[i] > 0:
                    ratio = genome.frequencies[j] / genome.frequencies[i]
                    for special_ratio in special_ratios:
                        error = abs(ratio - special_ratio) / special_ratio
                        if error < 0.1:  # Within 10%
                            harmonic_fitness += (1.0 - error) * 10
        
        # Spectral complexity fitness (not too simple, not too chaotic)
        spectral_entropy = 0.0
        if np.sum(power_spectrum) > 0:
            normalized_spectrum = power_spectrum / np.sum(power_spectrum)
            nonzero_spectrum = normalized_spectrum[normalized_spectrum > 1e-12]
            if len(nonzero_spectrum) > 0:
                spectral_entropy = -np.sum(nonzero_spectrum * np.log2(nonzero_spectrum))
        
        max_entropy = math.log2(len(power_spectrum))
        if max_entropy > 0:
            normalized_entropy = spectral_entropy / max_entropy
            # Sweet spot around 0.6-0.8
            if normalized_entropy < 0.6:
                complexity_fitness = normalized_entropy / 0.6 * 5
            elif normalized_entropy > 0.8:
                complexity_fitness = (1.0 - normalized_entropy) / 0.2 * 5
            else:
                complexity_fitness = 5.0
        else:
            complexity_fitness = 0.0
        
        # Amplitude balance fitness
        balance_fitness = 0.0
        total_amplitude = sum(genome.amplitudes)
        if total_amplitude > 0:
            balance = min(genome.amplitudes) / max(genome.amplitudes)
            balance_fitness = balance * 3  # Prefer balanced amplitudes
        
        # Age penalty (encourage exploration)
        age_penalty = min(genome.age * 0.1, 5.0)
        
        total_fitness = harmonic_fitness + complexity_fitness + balance_fitness - age_penalty
        return max(0.0, total_fitness)
    
    def evolve_generation(self, current_signal):
        """Evolve the population toward higher fitness"""
        
        # Evaluate all genomes
        for genome in self.population:
            genome.fitness = self.evaluate_fitness(genome, current_signal)
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Check for champion (high fitness threshold)
        if self.population[0].fitness > 15.0:
            if self.current_champion != self.population[0]:
                self.current_champion = self.population[0]
                self.lock_duration = 60  # Lock for 60 frames (~2 seconds at 30fps)
                return True  # Signal that we found a new champion
        
        # Reduce lock duration
        if self.lock_duration > 0:
            self.lock_duration -= 1
        
        # If locked, don't evolve
        if self.lock_duration > 0:
            return False
        
        # Create new population
        elite_count = max(1, self.population_size // 4)  # Keep top 25%
        new_population = self.population[:elite_count]
        
        # Fill rest with mutations and crossovers
        while len(new_population) < self.population_size:
            if random.random() < 0.7:  # 70% mutations
                parent = random.choice(self.population[:self.population_size//2])
                child = parent.mutate(strength=0.2 * self.exploration_pressure)
                new_population.append(child)
            else:  # 30% fresh random genomes for exploration
                genome = WaveGenome(
                    frequencies=[random.uniform(0.5, 8.0) for _ in range(3)],
                    amplitudes=[random.uniform(0.2, 1.2) for _ in range(3)],
                    phases=[random.uniform(0, 2*math.pi) for _ in range(3)],
                    modulation_rates=[random.uniform(-0.2, 0.2) for _ in range(3)]
                )
                new_population.append(genome)
        
        self.population = new_population
        self.generation += 1
        
        # Adjust exploration pressure
        if self.generation % 50 == 0:
            self.exploration_pressure *= 0.95  # Gradually reduce exploration
        
        return False

class LivingOscilloscope:
    def __init__(self, root):
        self.root = root
        self.root.title("Living Oscilloscope - Genetic Resonance Explorer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#000011')
        
        # Core systems
        self.wave_generator = GeneticWaveGenerator()
        self.running = False
        
        # Visual state
        self.signal_history = np.zeros((1000, 800))  # Rolling buffer for smooth scrolling
        self.history_index = 0
        self.discovered_patterns = []
        self.time_offset = 0.0
        
        # Performance
        self.last_update = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Thread communication
        self.update_queue = Queue()
        self.compute_thread = None
        
        self.setup_ui()
        self.start_evolution()
    
    def setup_ui(self):
        """Create the living instrument interface"""
        
        # Main display
        display_frame = tk.Frame(self.root, bg='#000011')
        display_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Large flowing oscilloscope
        self.canvas = tk.Canvas(display_frame, bg='#000022', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(display_frame, bg='#000011', height=140)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        control_frame.pack_propagate(False)
        
        # Left controls
        left_frame = tk.Frame(control_frame, bg='#000011')
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Power switch
        self.power_btn = tk.Button(left_frame, text="◉ POWER", font=('Courier', 16, 'bold'),
                                 bg='#003300', fg='#00ff00', width=10, height=3,
                                 command=self.toggle_power)
        self.power_btn.pack(pady=5)
        
        # Status display
        status_frame = tk.Frame(left_frame, bg='#001100', relief=tk.SUNKEN, bd=2)
        status_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.freq_label = tk.Label(status_frame, text="0.0 Hz", font=('Courier', 14, 'bold'),
                                 bg='#001100', fg='#00ff88')
        self.freq_label.pack()
        
        self.gen_label = tk.Label(status_frame, text="Gen: 0", font=('Courier', 10),
                                bg='#001100', fg='#00aa44')
        self.gen_label.pack()
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", font=('Courier', 10),
                                bg='#001100', fg='#00aa44')
        self.fps_label.pack()
        
        # Center - Evolution parameters
        evo_frame = tk.LabelFrame(control_frame, text="GENETIC PARAMETERS", 
                                bg='#000011', fg='#00ff00', font=('Courier', 10, 'bold'))
        evo_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        # Mutation rate
        tk.Label(evo_frame, text="MUTATION", bg='#000011', fg='#00ff00', 
                font=('Courier', 8)).pack()
        self.mutation_var = tk.DoubleVar(value=0.2)
        tk.Scale(evo_frame, from_=0.05, to=0.5, resolution=0.05, orient=tk.HORIZONTAL,
               variable=self.mutation_var, bg='#000011', fg='#00ff00', length=120).pack()
        
        # Population size display
        tk.Label(evo_frame, text="POPULATION", bg='#000011', fg='#00ff00',
                font=('Courier', 8)).pack()
        self.pop_label = tk.Label(evo_frame, text="20", font=('Courier', 12, 'bold'),
                                bg='#000011', fg='#ffff00')
        self.pop_label.pack()
        
        # Champion fitness
        tk.Label(evo_frame, text="CHAMPION", bg='#000011', fg='#00ff00',
                font=('Courier', 8)).pack()
        self.fitness_label = tk.Label(evo_frame, text="0.0", font=('Courier', 12, 'bold'),
                                    bg='#000011', fg='#ffff00')
        self.fitness_label.pack()
        
        # Right side - Pattern discovery log
        log_frame = tk.LabelFrame(control_frame, text="DISCOVERED RESONANCES",
                                bg='#000011', fg='#00ff00', font=('Courier', 10, 'bold'))
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        # Scrollable pattern log
        log_container = tk.Frame(log_frame, bg='#000011')
        log_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pattern_log = tk.Listbox(log_container, font=('Courier', 9),
                                    bg='#001100', fg='#00ff88',
                                    selectbackground='#003300',
                                    yscrollcommand=scrollbar.set)
        self.pattern_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.pattern_log.yview)
        
        # Lock indicator
        lock_frame = tk.Frame(control_frame, bg='#000011')
        lock_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Label(lock_frame, text="LOCK", bg='#000011', fg='#00ff00',
                font=('Courier', 12, 'bold')).pack()
        self.lock_indicator = tk.Label(lock_frame, text="●", font=('Courier', 24),
                                     bg='#000011', fg='#ff0000')
        self.lock_indicator.pack()
    
    def toggle_power(self):
        """Toggle the evolution system"""
        self.running = not self.running
        
        if self.running:
            self.power_btn.config(text="◉ POWER", bg='#330000', fg='#ff4444')
        else:
            self.power_btn.config(text="◉ POWER", bg='#003300', fg='#00ff00')
            self.lock_indicator.config(fg='#ff0000')
    
    def start_evolution(self):
        """Start the genetic evolution thread"""
        self.compute_thread = threading.Thread(target=self.evolution_loop, daemon=True)
        self.compute_thread.start()
        self.update_display()
    
    def evolution_loop(self):
        """Background evolution computation"""
        while True:
            if self.running:
                current_time = time.time()
                
                # Get current best genome
                best_genome = self.wave_generator.population[0] if self.wave_generator.population else None
                
                if best_genome:
                    # Generate signal from best genome
                    signal = self.generate_signal_from_genome(best_genome, current_time)
                    
                    # Evolve population
                    found_champion = self.wave_generator.evolve_generation(signal)
                    
                    # Queue update for main thread
                    self.update_queue.put({
                        'signal': signal,
                        'genome': best_genome,
                        'generation': self.wave_generator.generation,
                        'champion': found_champion,
                        'locked': self.wave_generator.lock_duration > 0,
                        'fitness': best_genome.fitness
                    })
                
            time.sleep(1/60)  # 60 FPS evolution
    
    def generate_signal_from_genome(self, genome, t, duration=0.5):
        """Generate wave signal from genetic parameters"""
        sample_rate = 1000
        samples = int(sample_rate * duration)
        time_points = np.linspace(0, duration, samples)
        
        signal = np.zeros(samples)
        
        for i, (freq, amp, phase, mod_rate) in enumerate(zip(
            genome.frequencies, genome.amplitudes, genome.phases, genome.modulation_rates)):
            
            # Add time-varying modulation
            modulated_freq = freq * (1 + 0.1 * math.sin(mod_rate * t))
            phase_offset = phase + i * math.pi / 4
            
            wave_component = amp * np.sin(2 * np.pi * modulated_freq * (time_points + t * 0.1) + phase_offset)
            signal += wave_component
        
        return signal
    
    def update_display(self):
        """Update the visual display"""
        current_time = time.time()
        
        # Process evolution updates
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                self.process_evolution_update(update)
            except:
                break
        
        # Draw the oscilloscope
        if self.running:
            self.draw_living_scope()
        
        # Update FPS
        self.frame_count += 1
        if current_time - self.last_update > 1.0:
            self.fps = self.frame_count / (current_time - self.last_update)
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.frame_count = 0
            self.last_update = current_time
        
        # Schedule next update
        self.root.after(16, self.update_display)  # ~60 FPS
    
    def process_evolution_update(self, update):
        """Process an evolution update"""
        
        # Add signal to history for scrolling display
        signal = update['signal']
        if len(signal) > 0:
            # Downsample signal to fit display width
            display_width = 800
            if len(signal) > display_width:
                indices = np.linspace(0, len(signal)-1, display_width, dtype=int)
                display_signal = signal[indices]
            else:
                display_signal = np.pad(signal, (0, display_width - len(signal)), 'constant')
            
            # Add to rolling history
            self.signal_history[self.history_index] = display_signal
            self.history_index = (self.history_index + 1) % len(self.signal_history)
        
        # Update status displays
        genome = update['genome']
        avg_freq = np.mean(genome.frequencies)
        self.freq_label.config(text=f"{avg_freq:.1f} Hz")
        self.gen_label.config(text=f"Gen: {update['generation']}")
        self.fitness_label.config(text=f"{update['fitness']:.1f}")
        
        # Update lock indicator
        if update['locked']:
            self.lock_indicator.config(fg='#ffff00')  # Yellow when locked
        else:
            self.lock_indicator.config(fg='#ff0000')  # Red when evolving
        
        # Log new champions
        if update['champion'] and update['fitness'] > 10.0:
            pattern_desc = f"Gen {update['generation']}: Fitness {update['fitness']:.1f} | "
            pattern_desc += f"Freqs: {[f'{f:.1f}' for f in genome.frequencies[:2]]}"
            self.pattern_log.insert(tk.END, pattern_desc)
            self.pattern_log.see(tk.END)
            
            # Keep log manageable
            if self.pattern_log.size() > 50:
                self.pattern_log.delete(0, 10)
    
    def draw_living_scope(self):
        """Draw the flowing, living oscilloscope display"""
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Draw subtle grid
        self.draw_scope_grid(width, height)
        
        # Draw flowing waveforms
        self.draw_flowing_waves(width, height)
        
        # Draw evolution visualization
        self.draw_evolution_status(width, height)
        
        # Draw frequency spectrum
        self.draw_spectrum_display(width, height)
    
    def draw_scope_grid(self, width, height):
        """Draw oscilloscope grid"""
        # Center crosshairs
        center_x, center_y = width // 2, height // 2
        self.canvas.create_line(0, center_y, width, center_y, fill='#003366', width=1)
        self.canvas.create_line(center_x, 0, center_x, height, fill='#003366', width=1)
        
        # Grid lines
        for i in range(1, 8):
            x = width * i / 8
            self.canvas.create_line(x, 0, x, height, fill='#001133', width=1)
        
        for i in range(1, 6):
            y = height * i / 6
            self.canvas.create_line(0, y, width, y, fill='#001133', width=1)
    
    def draw_flowing_waves(self, width, height):
        """Draw smooth flowing waveforms"""
        if len(self.signal_history) == 0:
            return
        
        center_y = height // 2
        max_amplitude = height // 2 - 40
        
        # Draw multiple recent traces for persistence effect
        trace_count = min(50, len(self.signal_history))
        
        for trace_age in range(trace_count):
            # Calculate history index (newer traces first)
            hist_idx = (self.history_index - trace_age - 1) % len(self.signal_history)
            signal_row = self.signal_history[hist_idx]
            
            if np.max(np.abs(signal_row)) == 0:
                continue
            
            # Fade older traces
            alpha = (trace_count - trace_age) / trace_count
            alpha = alpha ** 0.5  # Non-linear fade
            
            # Color based on age
            if alpha > 0.8:
                color = '#00ff88'  # Bright green for newest
            elif alpha > 0.5:
                color = '#00cc66'  # Medium green
            else:
                color = '#004422'  # Dark green for oldest
            
            # Normalize signal
            max_val = max(np.max(np.abs(signal_row)), 1e-6)
            normalized_signal = signal_row / max_val
            
            # Create smooth flowing line
            points = []
            for i, val in enumerate(normalized_signal):
                x = i * width / len(normalized_signal)
                y = center_y - val * max_amplitude * alpha
                points.extend([x, y])
            
            if len(points) >= 4:
                self.canvas.create_line(points, fill=color, width=max(1, int(alpha * 3)), 
                                      smooth=True)
    
    def draw_evolution_status(self, width, height):
        """Draw genetic evolution visualization"""
        if not hasattr(self.wave_generator, 'population') or not self.wave_generator.population:
            return
        
        # Top-left corner evolution display
        evo_x, evo_y = 20, 20
        evo_width, evo_height = 200, 80
        
        # Background
        self.canvas.create_rectangle(evo_x, evo_y, evo_x + evo_width, evo_y + evo_height,
                                   fill='#000033', outline='#0066cc', width=1)
        
        # Population fitness distribution
        population = sorted(self.wave_generator.population, key=lambda g: g.fitness, reverse=True)
        
        if len(population) > 0:
            max_fitness = max(g.fitness for g in population[:10])  # Top 10
            
            for i, genome in enumerate(population[:10]):
                if max_fitness > 0:
                    bar_height = (genome.fitness / max_fitness) * (evo_height - 20)
                    bar_x = evo_x + 10 + i * (evo_width - 20) / 10
                    bar_y = evo_y + evo_height - 10 - bar_height
                    
                    # Color based on fitness
                    if genome.fitness > 15:
                        color = '#ffff00'  # Gold for champions
                    elif genome.fitness > 10:
                        color = '#00ff00'  # Green for good
                    else:
                        color = '#0088cc'  # Blue for average
                    
                    self.canvas.create_rectangle(bar_x, bar_y, bar_x + 8, evo_y + evo_height - 10,
                                               fill=color, outline='')
        
        # Generation counter
        self.canvas.create_text(evo_x + 10, evo_y + 10, text=f"Gen: {self.wave_generator.generation}",
                              fill='#00ffff', font=('Courier', 10, 'bold'), anchor='nw')
    
    def draw_spectrum_display(self, width, height):
        """Draw real-time frequency spectrum"""
        if self.history_index == 0:
            return
        
        # Bottom-right spectrum analyzer
        spec_x = width - 250
        spec_y = height - 120
        spec_width = 230
        spec_height = 100
        
        # Background
        self.canvas.create_rectangle(spec_x, spec_y, spec_x + spec_width, spec_y + spec_height,
                                   fill='#000033', outline='#0066cc', width=1)
        
        # Get latest signal
        latest_signal = self.signal_history[(self.history_index - 1) % len(self.signal_history)]
        
        if len(latest_signal) > 0 and np.max(np.abs(latest_signal)) > 0:
            # FFT
            fft = np.fft.fft(latest_signal)
            power_spectrum = np.abs(fft[:len(fft)//2])
            
            if len(power_spectrum) > 0 and np.max(power_spectrum) > 0:
                power_spectrum = power_spectrum / np.max(power_spectrum)
                
                # Draw spectrum bars
                bar_count = min(50, len(power_spectrum))
                bar_width = (spec_width - 20) / bar_count
                
                for i in range(bar_count):
                    idx = int(i * len(power_spectrum) / bar_count)
                    val = power_spectrum[idx]
                    
                    if val > 0.05:  # Only draw significant components
                        bar_height = val * (spec_height - 20)
                        bar_x = spec_x + 10 + i * bar_width
                        bar_y = spec_y + spec_height - 10 - bar_height
                        
                        # Color based on intensity
                        if val > 0.8:
                            color = '#ff4444'  # Red for peaks
                        elif val > 0.5:
                            color = '#ffaa00'  # Orange for strong
                        else:
                            color = '#0088ff'  # Blue for moderate
                        
                        self.canvas.create_rectangle(bar_x, bar_y, bar_x + bar_width - 1, 
                                                   spec_y + spec_height - 10,
                                                   fill=color, outline='')

def main():
    root = tk.Tk()
    app = LivingOscilloscope(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\\nShutting down Living Oscilloscope...")

if __name__ == "__main__":
    main()