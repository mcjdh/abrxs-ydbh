#!/usr/bin/env python3
"""
Resonance Patterns
Exploring how simple oscillations create complex interference patterns
Like thoughts meeting thoughts, waves meeting waves
"""

import math
import time

def wave(x, t, freq, phase=0, amplitude=1):
    """A simple sine wave"""
    return amplitude * math.sin(2 * math.pi * freq * (x + t) + phase)

def interference(x, t, waves):
    """Sum of multiple waves - where resonance lives"""
    return sum(wave(x, t, w['freq'], w.get('phase', 0), w.get('amp', 1)) 
               for w in waves)

def visualize_wave(values, width=60, height=10):
    """Convert wave values to ASCII art"""
    # Normalize to display range
    max_val = max(abs(v) for v in values)
    if max_val == 0:
        return [' ' * width for _ in range(height)]
    
    normalized = [v / max_val for v in values]
    
    lines = []
    for row in range(height):
        line = ""
        y_level = (height/2 - row - 0.5) / (height/2)  # -1 to 1
        
        for i, val in enumerate(normalized):
            if abs(val - y_level) < 0.15:  # Close enough to this y level
                intensity = 1 - abs(val - y_level) / 0.15
                if intensity > 0.7:
                    line += '#'
                elif intensity > 0.4:
                    line += '+'
                elif intensity > 0.1:
                    line += '.'
                else:
                    line += ' '
            else:
                line += ' '
        lines.append(line)
    
    return lines

def main():
    print("Resonance: Where Waves Meet Waves")
    print("=" * 40)
    
    # Define some wave frequencies that create interesting patterns
    wave_sets = [
        {
            'name': 'Harmony (1:2 ratio)',
            'waves': [{'freq': 1}, {'freq': 2, 'amp': 0.7}]
        },
        {
            'name': 'Golden Ratio (1:phi)',
            'waves': [{'freq': 1}, {'freq': 1.618, 'amp': 0.8}]
        },
        {
            'name': 'Near Miss (creates beating)',
            'waves': [{'freq': 1}, {'freq': 1.05, 'amp': 1}]
        },
        {
            'name': 'Chaos (irrational ratios)',
            'waves': [{'freq': 1}, {'freq': math.pi/2, 'amp': 0.6}, {'freq': math.sqrt(2), 'amp': 0.4}]
        }
    ]
    
    width = 60
    x_points = [i / width * 4 for i in range(width)]  # 4 wavelengths
    
    for wave_set in wave_sets:
        print(f"\n{wave_set['name']}")
        print("-" * len(wave_set['name']))
        
        # Show a few frames of animation
        for frame in range(3):
            t = frame * 0.3
            values = [interference(x, t, wave_set['waves']) for x in x_points]
            lines = visualize_wave(values, width)
            
            for line in lines:
                print(line)
            print()
            
        print("Wave frequencies:", [w['freq'] for w in wave_set['waves']])
        print()
    
    print("When frequencies align, patterns emerge.")
    print("When they drift, complexity blooms.")
    print("In the space between order and chaos,")
    print("resonance finds its voice.")

if __name__ == "__main__":
    main()