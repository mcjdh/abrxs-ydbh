#!/usr/bin/env python3
"""
Cymatic Explorer - Automated Resonance Detection with Visual Interface
A minimal, radio-dial style interface for exploring the frequency space
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import time
import threading
from queue import Queue
from resonance_detector import ResonanceDetector, FrequencyScanner

class CymaticExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Cymatic Explorer - Resonance Detector")
        self.root.geometry("1200x800")
        self.root.configure(bg='#000011')
        
        # Detection system
        self.detector = ResonanceDetector()
        self.scanner = FrequencyScanner(self.detector)
        
        # Visual state
        self.running = False
        self.pattern_history = []
        self.signal_buffer = np.zeros(800)
        self.buffer_index = 0
        
        # Performance optimization
        self.last_draw_time = 0
        self.fps_limit = 30
        
        # Thread communication
        self.pattern_queue = Queue()
        self.detection_thread = None
        
        self.setup_ui()
        self.start_detection()
        
    def setup_ui(self):
        """Create minimal, radio-dial style interface"""
        
        # Main display area
        display_frame = tk.Frame(self.root, bg='#000011')
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Large oscilloscope display
        self.canvas = tk.Canvas(display_frame, bg='#001122', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(display_frame, bg='#000011', height=120)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))
        control_frame.pack_propagate(False)
        
        # Left side - Radio dial controls
        dial_frame = tk.Frame(control_frame, bg='#000011')
        dial_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Power button
        self.power_button = tk.Button(dial_frame, text="POWER", 
                                    command=self.toggle_power,
                                    bg='#003300', fg='#00ff00',
                                    font=('Courier', 14, 'bold'),
                                    width=8, height=2)
        self.power_button.pack(pady=10)
        
        # Frequency display
        freq_display_frame = tk.Frame(dial_frame, bg='#000011')
        freq_display_frame.pack(pady=10)
        
        tk.Label(freq_display_frame, text="FREQUENCY", 
                bg='#000011', fg='#00ff00', 
                font=('Courier', 8)).pack()
        
        self.freq_display = tk.Label(freq_display_frame, 
                                   text="0.00 Hz",
                                   bg='#001100', fg='#00ff00',
                                   font=('Courier', 16, 'bold'),
                                   width=12, relief=tk.SUNKEN)
        self.freq_display.pack()
        
        # Scan speed control
        speed_frame = tk.Frame(dial_frame, bg='#000011')
        speed_frame.pack(pady=10)
        
        tk.Label(speed_frame, text="SCAN SPEED", 
                bg='#000011', fg='#00ff00', 
                font=('Courier', 8)).pack()
        
        self.speed_var = tk.DoubleVar(value=0.1)
        self.speed_scale = tk.Scale(speed_frame, from_=0.01, to=1.0, resolution=0.01,
                                  orient=tk.HORIZONTAL, variable=self.speed_var,
                                  bg='#000011', fg='#00ff00', 
                                  highlightbackground='#000011', length=120)
        self.speed_scale.pack()
        
        # Pattern log
        log_frame = tk.Frame(control_frame, bg='#000011')
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        tk.Label(log_frame, text="DETECTED PATTERNS", 
                bg='#000011', fg='#00ff00', 
                font=('Courier', 10, 'bold')).pack()
        
        log_container = tk.Frame(log_frame, bg='#000011')
        log_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pattern_log = tk.Listbox(log_container, 
                                    bg='#001100', fg='#00ff00',
                                    font=('Courier', 8),
                                    yscrollcommand=scrollbar.set,
                                    selectbackground='#003300')
        self.pattern_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.pattern_log.yview)
        
        # Status indicators
        status_frame = tk.Frame(control_frame, bg='#000011')
        status_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        self.status_labels = {}
        
        # Scanning indicator
        scan_frame = tk.Frame(status_frame, bg='#000011')
        scan_frame.pack(pady=5)
        
        tk.Label(scan_frame, text="SCAN", bg='#000011', fg='#00ff00', 
                font=('Courier', 8)).pack()
        self.status_labels['scan'] = tk.Label(scan_frame, text="●", 
                                            bg='#000011', fg='#ff0000', 
                                            font=('Courier', 16))
        self.status_labels['scan'].pack()
        
        # Lock indicator
        lock_frame = tk.Frame(status_frame, bg='#000011')
        lock_frame.pack(pady=5)
        
        tk.Label(lock_frame, text="LOCK", bg='#000011', fg='#00ff00', 
                font=('Courier', 8)).pack()
        self.status_labels['lock'] = tk.Label(lock_frame, text="●", 
                                            bg='#000011', fg='#ff0000', 
                                            font=('Courier', 16))
        self.status_labels['lock'].pack()
        
        # Beauty meter
        beauty_frame = tk.Frame(status_frame, bg='#000011')
        beauty_frame.pack(pady=5)
        
        tk.Label(beauty_frame, text="BEAUTY", bg='#000011', fg='#00ff00', 
                font=('Courier', 8)).pack()
        self.beauty_meter = tk.Canvas(beauty_frame, width=30, height=100, 
                                    bg='#001100', highlightthickness=0)
        self.beauty_meter.pack()
        
    def toggle_power(self):
        """Toggle the scanner on/off"""
        self.running = not self.running
        
        if self.running:
            self.power_button.config(text="POWER", bg='#330000', fg='#ff0000')
            self.status_labels['scan'].config(fg='#00ff00')
        else:
            self.power_button.config(text="POWER", bg='#003300', fg='#00ff00')
            self.status_labels['scan'].config(fg='#ff0000')
            self.status_labels['lock'].config(fg='#ff0000')
    
    def start_detection(self):
        """Start the detection thread"""
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        # Start visual update loop
        self.update_display()
    
    def detection_loop(self):
        """Background thread for pattern detection"""
        while True:
            if self.running:
                # Update scan speed
                self.scanner.scan_speed = self.speed_var.get()
                
                # Update scanner
                pattern = self.scanner.update_scan()
                
                # Get current signal for display
                signal, frequencies = self.scanner.generate_signal(0.1)
                
                # Queue updates for main thread
                self.pattern_queue.put({
                    'signal': signal,
                    'frequencies': frequencies,
                    'pattern': pattern,
                    'locked': self.scanner.locked_pattern is not None
                })
            
            time.sleep(0.03)  # ~33 FPS detection rate
    
    def update_display(self):
        """Update the visual display"""
        current_time = time.time()
        
        # FPS limiting
        if current_time - self.last_draw_time < 1.0 / self.fps_limit:
            self.root.after(16, self.update_display)
            return
        
        # Process queued updates
        while not self.pattern_queue.empty():
            try:
                update = self.pattern_queue.get_nowait()
                self.process_update(update)
            except:
                break
        
        # Draw the oscilloscope display
        if self.running:
            self.draw_oscilloscope()
        
        self.last_draw_time = current_time
        self.root.after(16, self.update_display)
    
    def process_update(self, update):
        """Process an update from the detection thread"""
        # Update signal buffer
        signal = update['signal']
        chunk_size = min(len(signal), len(self.signal_buffer) // 4)
        
        if chunk_size > 0:
            end_idx = (self.buffer_index + chunk_size) % len(self.signal_buffer)
            if end_idx > self.buffer_index:
                self.signal_buffer[self.buffer_index:end_idx] = signal[:chunk_size]
            else:
                first_part = len(self.signal_buffer) - self.buffer_index
                self.signal_buffer[self.buffer_index:] = signal[:first_part]
                self.signal_buffer[:end_idx] = signal[first_part:chunk_size]
            
            self.buffer_index = end_idx
        
        # Update frequency display
        avg_freq = np.mean(update['frequencies'])
        self.freq_display.config(text=f"{avg_freq:.2f} Hz")
        
        # Update lock indicator
        if update['locked']:
            self.status_labels['lock'].config(fg='#ffff00')  # Yellow when locked
        else:
            self.status_labels['lock'].config(fg='#ff0000')  # Red when scanning
        
        # Log new patterns
        if update['pattern']:
            self.pattern_history.append(update['pattern'])
            
            # Add to visual log
            pattern_str = f"{len(self.pattern_history):02d}: {update['pattern'].pattern_type} ({update['pattern'].beauty_score:.3f})"
            self.pattern_log.insert(tk.END, pattern_str)
            self.pattern_log.see(tk.END)
            
            # Update beauty meter
            self.update_beauty_meter(update['pattern'].beauty_score)
    
    def draw_oscilloscope(self):
        """Draw the high-performance oscilloscope display"""
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Draw grid (minimal)
        self.draw_grid(width, height)
        
        # Draw waveform
        self.draw_waveform(width, height)
        
        # Draw frequency spectrum
        self.draw_spectrum(width, height)
    
    def draw_grid(self, width, height):
        """Draw minimal oscilloscope grid"""
        # Center lines
        self.canvas.create_line(0, height//2, width, height//2, fill='#003366', width=1)
        self.canvas.create_line(width//2, 0, width//2, height, fill='#003366', width=1)
        
        # Grid lines
        for i in range(1, 4):
            y = height * i / 4
            self.canvas.create_line(0, y, width, y, fill='#001122', width=1)
        
        for i in range(1, 8):
            x = width * i / 8
            self.canvas.create_line(x, 0, x, height, fill='#001122', width=1)
    
    def draw_waveform(self, width, height):
        """Draw the main waveform"""
        if len(self.signal_buffer) == 0:
            return
        
        # Get data from circular buffer
        ordered_signal = np.concatenate([
            self.signal_buffer[self.buffer_index:],
            self.signal_buffer[:self.buffer_index]
        ])
        
        # Downsample for display
        display_points = min(width, len(ordered_signal))
        if len(ordered_signal) > display_points:
            indices = np.linspace(0, len(ordered_signal)-1, display_points, dtype=int)
            display_signal = ordered_signal[indices]
        else:
            display_signal = ordered_signal
        
        # Convert to screen coordinates
        if len(display_signal) > 0:
            max_val = max(abs(np.max(display_signal)), abs(np.min(display_signal)), 1e-6)
            normalized_signal = display_signal / max_val
            
            center_y = height // 2
            scale = (height // 2 - 20)
            
            points = []
            for i, val in enumerate(normalized_signal):
                x = i * width / len(normalized_signal)
                y = center_y - val * scale
                points.extend([x, y])
            
            if len(points) >= 4:
                self.canvas.create_line(points, fill='#00ff88', width=2, smooth=True)
    
    def draw_spectrum(self, width, height):
        """Draw frequency spectrum in corner"""
        if len(self.signal_buffer) == 0:
            return
        
        spec_width = 200
        spec_height = 100
        spec_x = width - spec_width - 10
        spec_y = 10
        
        # Background
        self.canvas.create_rectangle(spec_x, spec_y, spec_x + spec_width, spec_y + spec_height,
                                   fill='#000022', outline='#003366')
        
        # Get recent signal for FFT
        recent_signal = self.signal_buffer[max(0, self.buffer_index-256):self.buffer_index]
        if len(recent_signal) > 32:
            # FFT
            fft = np.fft.fft(recent_signal)
            power_spectrum = np.abs(fft[:len(fft)//2])
            
            if len(power_spectrum) > 0 and np.max(power_spectrum) > 0:
                power_spectrum = power_spectrum / np.max(power_spectrum)
                
                # Draw spectrum bars
                bar_width = spec_width / len(power_spectrum)
                for i, val in enumerate(power_spectrum):
                    if val > 0.1:
                        bar_height = val * spec_height
                        x1 = spec_x + i * bar_width
                        y1 = spec_y + spec_height - bar_height
                        y2 = spec_y + spec_height
                        
                        self.canvas.create_rectangle(x1, y1, x1 + bar_width, y2,
                                                   fill='#0088ff', outline='')
    
    def update_beauty_meter(self, beauty_score):
        """Update the beauty meter display"""
        self.beauty_meter.delete("all")
        
        meter_height = 100
        meter_width = 30
        
        # Background
        self.beauty_meter.create_rectangle(0, 0, meter_width, meter_height,
                                         fill='#001100', outline='#003300')
        
        # Beauty level
        fill_height = int(beauty_score * meter_height)
        if fill_height > 0:
            if beauty_score < 0.3:
                color = '#ff0000'
            elif beauty_score < 0.7:
                color = '#ffff00'
            else:
                color = '#00ff00'
            
            self.beauty_meter.create_rectangle(2, meter_height - fill_height, 
                                             meter_width - 2, meter_height - 2,
                                             fill=color, outline='')
        
        # Scale marks
        for i in range(5):
            y = meter_height - (i * meter_height / 4)
            self.beauty_meter.create_line(0, y, 5, y, fill='#003300')

def main():
    root = tk.Tk()
    app = CymaticExplorer(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\\nExiting Cymatic Explorer...")

if __name__ == "__main__":
    main()