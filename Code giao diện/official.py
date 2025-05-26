
import tkinter as tk
from tkinter import ttk
import random
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import io
from PIL import Image
class SimulatedAnnealingDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulated Annealing Demo")
        self.root.geometry("900x600")
        self.root.configure(bg="white")
        
        # Set up the main container
        self.main_frame = tk.Frame(self.root, bg="white", padx=10, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame, 
            text="Simulated Annealing Demo", 
            font=("Arial", 24, "bold"),
            bg="white",
            fg="black",
        )
        self.title_label.pack(pady=1, padx=10, fill=tk.X)
        
        # Create content layout - left for parameters, right for visualization
        self.content_frame = tk.Frame(self.main_frame, bg="white")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx = (0, 0), pady=(20, 0))
        
        # Create the parameters frame (left side)
        self.params_frame = tk.Frame(self.content_frame, bg="red", width=300)
        self.params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=(20, 20))
        
        # Create the visualization frame (right side)
        self.visual_frame = tk.Frame(self.content_frame, bg="#e0e0e0", bd=0)
        self.visual_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create parameter inputs
        self.create_parameter_section()
        
        # Create visualization area
        # self.create_visualization()
        
    def create_parameter_section(self):
        # Parameters header
        parameters_header = tk.Label(
            self.params_frame,
            text="Parameters",
            font=("Arial", 16),
            bg="#c5cae9",  # Light purple background
            fg ="black",
            width=25,
            height=2
        )
        parameters_header.pack(fill=tk.X, pady=(20, 24))
        
        # Parameters input fields
        param_labels = [
            "Numbers of Cities",
            "Initial Temperature",
            "Cooling Rate",
            "Maximum Iteration",
            "Stopping Temperature"
        ]
        
        self.param_entries = {}
        
        for label_text in param_labels:
            frame = tk.Frame(self.params_frame, bg="white", pady=10)
            frame.pack(fill=tk.X)
            
            label = tk.Label(
                frame,
                text=label_text,
                font=("Arial", 12),
                bg="white",
                anchor="w",
                fg="black",
            )
            label.pack(side=tk.LEFT)
            
            entry = tk.Entry(
                frame,
                font=("Arial", 12),
                fg = "black",
                bg="#f0f0f0",
                relief=tk.SOLID,
                bd = 0.01
            )
            entry.pack(side=tk.RIGHT, padx=(10, 0), ipadx=10, ipady=5)
            
            self.param_entries[label_text] = entry
        
        # Set default values
        self.param_entries["Numbers of Cities"].insert(0, "20")
        self.param_entries["Initial Temperature"].insert(0, "1000.0")
        self.param_entries["Cooling Rate"].insert(0, "0.95")
        self.param_entries["Maximum Iteration"].insert(0, "1000")
        self.param_entries["Stopping Temperature"].insert(0, "1.0")
        
        # Apply button
        apply_button = tk.Button(
            self.params_frame,
            text="Apply",
            font=("Arial", 14, "bold"),
            bg="purple",  # Light purple background
            fg="black",
            bd=0,
            padx=10,
            pady=10,
            # command=self.run_simulation
        )
        apply_button.pack(fill=tk.X, pady=(20, 0))


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatedAnnealingDemo(root)
    root.mainloop()