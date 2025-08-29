#!/usr/bin/env python3
"""
Rule 30 Cellular Automaton
A simple rule that creates complex, beautiful patterns
Each cell's next state depends on itself and its two neighbors
"""

def rule30(left, center, right):
    """Rule 30: XOR of left and (center OR right)"""
    return left ^ (center | right)

def evolve_generation(cells):
    """Evolve one generation using Rule 30"""
    new_cells = [0] * len(cells)
    for i in range(len(cells)):
        left = cells[i-1] if i > 0 else 0
        center = cells[i]
        right = cells[i+1] if i < len(cells)-1 else 0
        new_cells[i] = rule30(left, center, right)
    return new_cells

def display_cells(cells):
    """Display cells as # and ."""
    return ''.join('#' if cell else '.' for cell in cells)

def main():
    width = 79  # Terminal width
    generations = 40
    
    # Start with a single active cell in the center
    cells = [0] * width
    cells[width // 2] = 1
    
    print("Rule 30 Cellular Automaton")
    print("Simple rules -> Complex patterns")
    print("-" * width)
    
    for gen in range(generations):
        print(display_cells(cells))
        cells = evolve_generation(cells)
    
    print("-" * width)
    print("Beauty from simplicity")

if __name__ == "__main__":
    main()