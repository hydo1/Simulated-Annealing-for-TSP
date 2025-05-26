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

# ---- Generate Random Cities ----
def generate_cities(num_cities):
    return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_cities)]
#random 2 số thập phân trong khoảng 0 - 10 để làm tọa độ x, y cho các thành phố


# ---- Distance Calculation ----
def distance(city1, city2):
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])
#tính độ dài cạnh huyền dựa trên x và y (x và y là cạng vuông dọc theo trục tọa độ)


def total_distance(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))
# khi i chạy tới tour cuối cùng, ví dụ 7 thì 7 % 7 = 0 quay về city đầu, nếu ko thì 2 % 7 = 2, 3 % 7= 3 ,...


# ---- 2-opt Swap ----
def two_opt(tour):
    a, b = sorted(random.sample(range(len(tour)), 2))
    new_tour = tour[:a] + tour[a:b+1][::-1] + tour[b+1:]
    return new_tour


# ---- Simulated Annealing Algorithm ----
def simulated_annealing(num_cities, initial_temp=1000, cooling_rate=0.995, stopping_temp=1e-8, max_iter=100000):
    cities = generate_cities(num_cities)
    n = len(cities)
    current_tour = list(range(n)) #list = 0,1,2,.. = số thành phố
    random.shuffle(current_tour) #trộn list
    current_cost = total_distance(current_tour, cities)
    best_tour = current_tour[:]
    best_cost = current_cost
    temp = initial_temp


    for i in range(max_iter):
        if temp < stopping_temp:
            break


        new_tour = two_opt(current_tour)
        new_cost = total_distance(new_tour, cities)
        delta = new_cost - current_cost


        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = new_tour
            current_cost = new_cost
            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost


        temp *= cooling_rate


    return best_tour, best_cost, cities

def plot_tour(tour, cities, filename="result.png"):
    tour_cities = [cities[i] for i in tour] + [cities[tour[0]]]
    xs, ys = zip(*tour_cities)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o')
    for idx, (x, y) in enumerate(cities):
        plt.text(x, y, str(idx), fontsize=12, ha='right', va='bottom')
    plt.title('TSP Tour (Simulated Annealing + 2-opt)')
    plt.grid(True)

    plt.savefig(filename)  # Save image here
    # plt.show()
    plt.close()





# ---- GUI Setup ----

class SimulatedAnnealingDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulated Annealing Demo")
        self.root.geometry("1200x650")
        self.root.configure(bg="white")
        self.image = None
        self.image_frame = None
        
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
        self.params_frame = tk.Frame(self.content_frame, bg="white", width=300)
        self.params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=(50, 50))
        
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
            font=("Arial", 18),
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
                font=("Arial", 14),
                bg="white",
                anchor="w",
                fg="black",
            )
            label.pack(side=tk.LEFT)
            
            entry = tk.Entry(
                frame,
                font=("Arial", 14),
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
            font=("Arial", 18),
            bg="purple",  # Light purple background
            fg="black",
            bd=0,
            padx=10,
            pady=10,
            command=self.run_simulation
        )
        apply_button.pack(fill=tk.X, pady=(20, 0))
        
    def run_simulation(self):
        """Run the simulated annealing algorithm and update the visualization"""
        
        # Get parameters from input fields
        num_cities = int(self.param_entries["Numbers of Cities"].get())
        initial_temp = float(self.param_entries["Initial Temperature"].get())
        cooling_rate = float(self.param_entries["Cooling Rate"].get())
        max_iterations = int(self.param_entries["Maximum Iteration"].get())
        stopping_temp = float(self.param_entries["Stopping Temperature"].get())

        best_tour, best_cost, cities = simulated_annealing(num_cities, initial_temp, cooling_rate, stopping_temp, max_iterations)

        self.plot(best_tour, cities)
        
    def plot(self, tour, cities):
        # Visualize the results
        # Call the function and save to file
        plot_tour(tour, cities, filename="result.png")

        self.image = tk.PhotoImage(file = "result.png")

        # Destroy old frame if it exists
        if self.image_frame is not None:
            self.image_frame.destroy()

        # Image frame
        self.image_frame = tk.Frame(self.visual_frame, width=400, height=400, bg="white")
        self.image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # tk.Label(
        #     image_frame,
        #     text="Edited Image",
        #     bg="grey",
        #     fg="white",
        # ).pack(padx=5, pady=5)

        tk.Label(self.image_frame, image = self.image).pack(padx=0, pady=0)
            
       


root = tk.Tk()
app = SimulatedAnnealingDemo(root)
root.mainloop()