#!/usr/bin/env python3
"""
Digital Oscilloscope - Visualizing Wave Resonance
Real-time exploration of how waves interact and create patterns
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import threading

class DigitalOscilloscope:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Oscilloscope - Wave Resonance Explorer")
        self.root.geometry("1000x700")
        self.root.configure(bg='#001100')
        
        # Oscilloscope state
        self.running = False
        self.time_offset = 0
        self.traces = []
        
        # Wave parameters
        self.waves = [
            {'freq': 1.0, 'amp': 1.0, 'phase': 0.0, 'enabled': True, 'color': '#00ff00'},
            {'freq': 2.0, 'amp': 0.7, 'phase': 0.0, 'enabled': True, 'color': '#ff6600'},
            {'freq': 1.618, 'amp': 0.8, 'phase': 0.0, 'enabled': False, 'color': '#ffff00'},
        ]
        
        self.setup_ui()
        self.animate()
    
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#001100')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for oscilloscope display
        self.canvas = tk.Canvas(main_frame, bg='#002200', width=800, height=400)
        self.canvas.pack(side=tk.TOP, pady=(0, 10))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#001100')
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Wave controls
        for i, wave in enumerate(self.waves):
            wave_frame = tk.LabelFrame(control_frame, text=f"Wave {i+1}", 
                                     bg='#001100', fg=wave['color'], 
                                     font=('Courier', 10, 'bold'))
            wave_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
            
            # Enable checkbox
            wave['enabled_var'] = tk.BooleanVar(value=wave['enabled'])
            tk.Checkbutton(wave_frame, text="Enable", variable=wave['enabled_var'],
                         bg='#001100', fg=wave['color'], selectcolor='#003300').pack()
            
            # Frequency slider
            tk.Label(wave_frame, text="Frequency", bg='#001100', fg=wave['color']).pack()
            wave['freq_var'] = tk.DoubleVar(value=wave['freq'])
            freq_scale = tk.Scale(wave_frame, from_=0.1, to=5.0, resolution=0.1,
                                orient=tk.HORIZONTAL, variable=wave['freq_var'],
                                bg='#001100', fg=wave['color'], highlightbackground='#001100')
            freq_scale.pack()
            
            # Amplitude slider
            tk.Label(wave_frame, text="Amplitude", bg='#001100', fg=wave['color']).pack()
            wave['amp_var'] = tk.DoubleVar(value=wave['amp'])
            amp_scale = tk.Scale(wave_frame, from_=0.0, to=2.0, resolution=0.1,
                               orient=tk.HORIZONTAL, variable=wave['amp_var'],
                               bg='#001100', fg=wave['color'], highlightbackground='#001100')
            amp_scale.pack()
            
            # Phase slider
            tk.Label(wave_frame, text="Phase", bg='#001100', fg=wave['color']).pack()
            wave['phase_var'] = tk.DoubleVar(value=wave['phase'])
            phase_scale = tk.Scale(wave_frame, from_=0, to=360, resolution=10,
                                 orient=tk.HORIZONTAL, variable=wave['phase_var'],
                                 bg='#001100', fg=wave['color'], highlightbackground='#001100')
            phase_scale.pack()
        
        # Global controls
        global_frame = tk.LabelFrame(control_frame, text="Global Controls", 
                                   bg='#001100', fg='#ffffff', 
                                   font=('Courier', 10, 'bold'))
        global_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.Y)
        
        # Start/Stop button
        self.start_button = tk.Button(global_frame, text="Start", 
                                    command=self.toggle_animation,
                                    bg='#003300', fg='#00ff00',
                                    font=('Courier', 12, 'bold'))
        self.start_button.pack(pady=5)
        
        # Preset buttons
        preset_frame = tk.Frame(global_frame, bg='#001100')
        preset_frame.pack()
        
        presets = [
            ("Harmony", [{'freq': 1.0, 'amp': 1.0}, {'freq': 2.0, 'amp': 0.7}, {'freq': 0.5, 'amp': 0.5}]),
            ("Golden", [{'freq': 1.0, 'amp': 1.0}, {'freq': 1.618, 'amp': 0.8}, {'freq': 0.618, 'amp': 0.6}]),
            ("Beats", [{'freq': 1.0, 'amp': 1.0}, {'freq': 1.05, 'amp': 1.0}, {'freq': 1.1, 'amp': 0.8}]),
        ]
        
        for name, values in presets:
            btn = tk.Button(preset_frame, text=name, 
                          command=lambda v=values: self.load_preset(v),
                          bg='#003300', fg='#00ff00', font=('Courier', 8))
            btn.pack(side=tk.LEFT, padx=2)
    
    def wave_function(self, x, t, freq, amp, phase):
        """Calculate wave value at position x and time t"""
        return amp * math.sin(2 * math.pi * freq * (x + t) + math.radians(phase))
    
    def update_wave_params(self):
        """Update wave parameters from UI controls"""
        for i, wave in enumerate(self.waves):
            wave['enabled'] = wave['enabled_var'].get()
            wave['freq'] = wave['freq_var'].get()
            wave['amp'] = wave['amp_var'].get()
            wave['phase'] = wave['phase_var'].get()
    
    def draw_scope(self):
        """Draw the oscilloscope display"""
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Draw grid
        self.draw_grid(width, height)
        
        # Update parameters
        self.update_wave_params()
        
        # Calculate time points
        x_points = [i / width * 4 for i in range(width)]  # 4 wavelengths across screen
        
        # Draw individual waves and combined signal
        combined_values = [0] * width
        
        for wave in self.waves:
            if not wave['enabled']:
                continue
                
            values = []
            for x in x_points:
                val = self.wave_function(x, self.time_offset, 
                                       wave['freq'], wave['amp'], wave['phase'])
                values.append(val)
                
            # Add to combined signal
            for i, val in enumerate(values):
                combined_values[i] += val
            
            # Draw individual wave (faded)
            self.draw_wave(values, width, height, wave['color'], alpha=0.3)
        
        # Draw combined wave (bright)
        self.draw_wave(combined_values, width, height, '#ffffff', alpha=1.0)
        
        # Draw labels
        self.canvas.create_text(10, 10, text=f"Time: {self.time_offset:.2f}s", 
                              fill='#00ff00', anchor='nw', font=('Courier', 10))
    
    def draw_grid(self, width, height):
        """Draw oscilloscope grid"""
        # Horizontal lines
        for i in range(5):
            y = height * i / 4
            color = '#004400' if i == 2 else '#002200'  # Center line brighter
            self.canvas.create_line(0, y, width, y, fill=color, width=1)
        
        # Vertical lines
        for i in range(9):
            x = width * i / 8
            color = '#004400' if i == 4 else '#002200'  # Center line brighter
            self.canvas.create_line(x, 0, x, height, fill=color, width=1)
    
    def draw_wave(self, values, width, height, color, alpha=1.0):
        """Draw a wave on the canvas"""
        if not values:
            return
            
        # Normalize values to fit canvas
        max_val = max(max(values), abs(min(values)), 1e-6)
        center_y = height / 2
        scale = (height / 2 - 10) / max_val
        
        points = []
        for i, val in enumerate(values):
            x = i
            y = center_y - val * scale
            points.extend([x, y])
        
        if len(points) >= 4:
            # Adjust color for alpha effect
            if alpha < 1.0:
                # Make color darker for alpha effect
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    r = int(r * alpha)
                    g = int(g * alpha)
                    b = int(b * alpha)
                    color = f'#{r:02x}{g:02x}{b:02x}'
            
            self.canvas.create_line(points, fill=color, width=2, smooth=True)
    
    def load_preset(self, values):
        """Load preset wave configuration"""
        for i, preset_wave in enumerate(values):
            if i < len(self.waves):
                self.waves[i]['freq_var'].set(preset_wave['freq'])
                self.waves[i]['amp_var'].set(preset_wave['amp'])
                self.waves[i]['enabled_var'].set(True)
    
    def toggle_animation(self):
        """Start/stop the animation"""
        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")
    
    def animate(self):
        """Animation loop"""
        if self.running:
            self.time_offset += 0.05
            self.draw_scope()
        
        # Schedule next frame
        self.root.after(50, self.animate)  # ~20 FPS

def main():
    root = tk.Tk()
    app = DigitalOscilloscope(root)
    root.mainloop()

if __name__ == "__main__":
    main()