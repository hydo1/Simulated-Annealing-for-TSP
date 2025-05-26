from ctypes import alignment
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
import giahy_code as giahy
import quangdai_code as quangdai
import math
import random
import matplotlib.pyplot as plt
import time
import os
import webbrowser


#=================UI=========================
class SimulatedAnnealingDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulated Annealing Demo")
        self.root.geometry("1400x820")
        self.root.configure(bg="white")
        self.image = None
        self.image_frame = None
        
        # Set up the main container
        self.main_frame = tk.Frame(self.root, bg="white", padx=10, pady=0)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame, 
            text="Simulated Annealing Demo", 
            font=("Arial", 28, "bold"),
            bg="white",
            fg="black",
        )
        self.title_label.pack(pady=1, padx=10, fill=tk.X)
        
        # Create content layout - left for parameters, right for visualization
        self.content_frame = tk.Frame(self.main_frame, bg="white")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx = (0, 0), pady=(20, 0))
        
        # Create the parameters frame (left side)
        self.Notebook_frame = ttk.Notebook(self.content_frame, width=300)
        self.Notebook_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=(20, 20))
        
        # Create the visualization frame (right side)
        self.visual_frame = tk.Frame(self.content_frame, bg="#e0e0e0", bd=0)
        self.visual_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=(0, 10))
        self.Notebook_frame.bind("<<NotebookTabChanged>>", self.on_tab_change)
        
        # Create parameter inputs
        self.create_giahy_tab()
        self.create_quangdai_tab()
        
        # Create visualization area
        # self.create_visualization()
    
    def on_tab_change(self, event):
            # Clear all widgets inside visual_frame
            for widget in self.visual_frame.winfo_children():
                widget.destroy()

    def create_giahy_tab(self):
        # create two tabs in notebook
        self.tab1 = tk.Frame(self.Notebook_frame, bg="white")
        
        self.Notebook_frame.add(self.tab1, text="Basic TSP")
        

        # customerize tab1
        #self.tab1.config(bg="red")

        # Parameters header
        toa_do = tk.Label(
            self.tab1,
            text="Add Cities",
            font=("Arial", 18, "bold"),
            bg="white",  # Light purple background
            fg ="black",
            width=25,
            height=1
        )
        toa_do.pack(fill=tk.X, pady=(0, 12))

        button_frame = tk.Frame(self.tab1, bg="white", pady=0)
        button_frame.pack(fill=tk.X)

        # Configure columns to expand evenly
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)

        # Entry fields for X and Y
        entry_x = tk.Entry(button_frame, width=6, font=("Helvetica", 14), bg="white", fg="black")
        entry_y = tk.Entry(button_frame, width=6, font=("Helvetica", 14), bg="white", fg="black")
        entry_x.insert(0, "1")
        entry_y.insert(0, "3")
        entry_x.config(justify="center")
        entry_y.config(justify="center")

        
        self.list_of_coordinates = []

        def add_coordinates_and_plot(x, y):
            try:
                x = float(x)
                y = float(y)
                self.list_of_coordinates.append((x, y))
                print(f"Coordinates added: {x}, {y}")
                entry_x.delete(0, tk.END)
                entry_y.delete(0, tk.END)
            except ValueError:
                print("Invalid input. Please enter numeric values.")

            # Function to plot the coordinates
            if self.list_of_coordinates:
                xs, ys = zip(*self.list_of_coordinates)
                plt.figure(figsize=(10, 6))
                plt.scatter(xs, ys)
                # plt.title('Coordinates Plot')
                # plt.xlabel('X-axis')
                # plt.ylabel('Y-axis')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("coordinates_plot.png")  # Save image here
                plt.close()
                # Load the image and display it
                self.image = tk.PhotoImage(file="coordinates_plot.png")

                # Destroy old frame if it exists
                if self.image_frame is not None:
                    self.image_frame.destroy()
                
                # Image frame
                self.image_frame = tk.Frame(self.visual_frame, width=400, height=400, bg="white")
                self.image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
                tk.Label(self.image_frame, image=self.image).pack(padx=0, pady=40)
                


        # Apply button
        btn_add = tk.Button(
            button_frame,
            text="Add",
            font=("Arial", 18),
            bg="purple",  # Light purple background
            fg="black",
            bd=0,
            padx=10,
            pady=0,
            command= lambda: add_coordinates_and_plot(entry_x.get(), entry_y.get())
        )
        
        # Layout widgets side by side
        entry_x.grid(row=0, column=0, padx=(4, 10))
        entry_y.grid(row=0, column=1, padx=(2, 10))
        btn_add.grid(row=0, column=2, padx=(6, 2))

        # Parameters header
        parameters_header = tk.Label(
            self.tab1,
            text="Set Parameters",
            font=("Arial", 18, "bold"),
            bg="white",  # Light purple background
            fg ="black",
            width=25,
            height=2
        )
        parameters_header.pack(fill=tk.X, pady=(0, 0))
        
        # Parameters input fields
        param_labels = [
            "Numbers of Cities",
            "Initial Temperature",
            "Cooling Rate",
            "Maximum Iteration",
            "Stopping Temperature",
            "Neighbor Method",
            "Initialization Method",
            "Cooling Strategy",
            "Only Nearest Neighbor",
        ]
        
        self.param_entries = {}
        
        for label_text in param_labels[:5]:
            frame = tk.Frame(self.tab1, bg="white", pady=10)
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
                bd = 0.01,
                width=10
            )
            entry.pack(side=tk.RIGHT, padx=(0, 4), ipadx=10, ipady=5)
            
            self.param_entries[label_text] = entry
        
        # Set default values
        self.param_entries["Numbers of Cities"].insert(0, "20")
        self.param_entries["Initial Temperature"].insert(0, "1000.0")
        self.param_entries["Cooling Rate"].insert(0, "0.995")
        self.param_entries["Maximum Iteration"].insert(0, "200000")
        self.param_entries["Stopping Temperature"].insert(0, "1e-200")

        neighbor_methods = ["two opt", "three opt", 'random swap']
        frame = tk.Frame(self.tab1, bg="white", pady=10)
        frame.pack(fill=tk.X)
        label = tk.Label(
            frame,
            text="Neighbor Method",
            font=("Arial", 14),
            bg="white",
            anchor="w",
            fg="black",
        )
        label.pack(side=tk.LEFT)
        neighbor_get = tk.StringVar(value=neighbor_methods[0])
        dropdown_neighbor = tk.OptionMenu(
            frame,
            neighbor_get,
            *neighbor_methods
        )
        dropdown_neighbor.pack(padx=(0, 4), side=tk.RIGHT)
        dropdown_neighbor.config(bd = 0, relief=tk.SOLID, width=8, bg="white", font=("Arial", 14), fg="black", anchor="w")
        self.param_entries["Neighbor Method"] = neighbor_get


        initialization_methods = ["random", "nn"]
        frame = tk.Frame(self.tab1, bg="white", pady=10)
        frame.pack(fill=tk.X)
        label = tk.Label(
            frame,
            text="Initialization Method",
            font=("Arial", 14),
            bg="white",
            anchor="w",
            fg="black",
        )
        label.pack(side=tk.LEFT)
        initialization_get = tk.StringVar(value=initialization_methods[0])
        dropdown_initialization = tk.OptionMenu(
            frame,
            initialization_get,
            *initialization_methods
        )
        dropdown_initialization.pack(padx=(0, 4), side=tk.RIGHT)
        dropdown_initialization.config(bd = 0, relief=tk.SOLID, width=8, bg="white", font=("Arial", 14), fg="black", anchor="w")
        self.param_entries["Initialization Method"] = initialization_get


        cooling_strategies = ["linear", "exponential", "logarithmic"]
        frame = tk.Frame(self.tab1, bg="white", pady=10)
        frame.pack(fill=tk.X)
        label = tk.Label(
            frame,
            text="Cooling Strategy",
            font=("Arial", 14),
            bg="white",
            anchor="w",
            fg="black",
        )
        label.pack(side=tk.LEFT)
        cooling_get = tk.StringVar(value=cooling_strategies[1])
        dropdown_cooling = tk.OptionMenu(
            frame,
            cooling_get,
            *cooling_strategies
        )
        dropdown_cooling.pack(padx=(0, 4), side=tk.RIGHT)
        dropdown_cooling.config(bd = 0, relief=tk.SOLID, width=8, bg="white", font=("Arial", 14), fg="black", anchor="w")
        self.param_entries["Cooling Strategy"] = cooling_get


        only_nearest_neighbors = tk.BooleanVar(value=False)
        frame = tk.Frame(self.tab1, bg="white", pady=10)
        frame.pack(fill=tk.X)
        label = tk.Label(
            frame,
            text="Only Nearest Neighbor",
            font=("Arial", 14),
            bg="white",
            anchor="w",
            fg="black",
        )
        label.pack(side=tk.LEFT)
        only_nearest_neighbors_checkbox = tk.Checkbutton(
            frame,
            variable=only_nearest_neighbors,
            bg="white",
            fg="black",
        )
        only_nearest_neighbors_checkbox.pack(padx=(0, 45), side=tk.RIGHT)
        self.param_entries["Only Nearest Neighbor"] = only_nearest_neighbors

        # create apply button
        apply_button = tk.Button(
            self.tab1,
            text="Apply",
            font=("Arial", 14, "bold"),
            bg="purple",  # Light purple background
            fg="black",
            bd=0,
            padx=1,
            pady=1,
            command= self.run_simulation_giahy
        )
        apply_button.pack(fill=tk.X, pady=(5, 0))

        # Create reset button
        reset_button = tk.Button(
            self.tab1,
            text="Clear Cities",
            font=("Arial", 14, "bold"),
            bg="purple",  # Light purple background
            fg="black",
            bd=0,
            padx=10,
            pady=10,
            command=self.reset_fields)
        reset_button.pack(fill=tk.X, pady=(5, 0))
        # Reset button function
    def reset_fields(self):
        '''Reset coordinates and image'''
        self.list_of_coordinates = []

        # Destroy old frame if it exists
        if self.image_frame is not None:
            self.image_frame.destroy()


    def create_quangdai_tab(self):
        self.tab2 = tk.Frame(self.Notebook_frame, bg="white")
        self.Notebook_frame.add(self.tab2, text="Advanced TSP")

        # Parameters header
        toa_do = tk.Label(
            self.tab2,
            text="Set Parameters",
            font=("Arial", 18, "bold"),
            # bg="#c5cae9",  # Light purple background
            fg ="black",
            bg="white",
            width=25,
            height=2
        )
        toa_do.pack(fill=tk.X, pady=(0, 0))

        params_list = [
            'Random Seed',
            'Number of Customers',
            'Number of Vehicles',
            'Initial Temperature',
            'Cooling rate',
            'Maximum iteration',
        ]

        self.param_entries_quangdai = {}
        for param in params_list:
            frame = tk.Frame(self.tab2, bg="white", pady=10)
            frame.pack(fill=tk.X)

            label = tk.Label(
                frame,
                text=param,
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
                bd = 0.01,
                width=10
            )
            entry.pack(side=tk.RIGHT, padx=(0, 4), ipadx=10, ipady=5)

            self.param_entries_quangdai[param] = entry
        
        # Set default values
        self.param_entries_quangdai["Random Seed"].insert(0, "42")
        self.param_entries_quangdai["Number of Customers"].insert(0, "20")
        self.param_entries_quangdai["Number of Vehicles"].insert(0, "5")
        self.param_entries_quangdai["Initial Temperature"].insert(0, "1000.0")
        self.param_entries_quangdai["Cooling rate"].insert(0, "0.95")
        self.param_entries_quangdai["Maximum iteration"].insert(0, "1000")

        # Dropdown list
        dropdown_list_params = {
            'Initialization Method' : ['random', 'nn'],
            'Cooling Strategy' : ['linear', 'exponential', 'logarithmic'],
            'Neighbor Method' : ['swap within route', 'two opt within route', 'swap between routes', 'move customer', 'random']
        }
        for param, options in dropdown_list_params.items():
            param_methods = options
            frame = tk.Frame(self.tab2, bg="white", pady=10)
            frame.pack(fill=tk.X)
            label = tk.Label(
                frame,
                text=param,
                font=("Arial", 14),
                bg="white",
                anchor="w",
                fg="black",
            )
            label.pack(side=tk.LEFT)
            value_get = tk.StringVar(value=param_methods[0])
            dropdown_neighbor = tk.OptionMenu(
                frame,
                value_get,
                *param_methods
            )
            dropdown_neighbor.pack(padx=(0, 4), side=tk.RIGHT)

            if param == 'Neighbor Method':
                dropdown_neighbor.config(bd = 0, relief=tk.SOLID, width=15, bg="white", font=("Arial", 14), fg="black", anchor="w")
            else:
                dropdown_neighbor.config(bd = 0, relief=tk.SOLID, width=8, bg="white", font=("Arial", 14), fg="black", anchor="w")
        
            self.param_entries_quangdai[param] = value_get
        
        # Checkbox list
        checkbox_list = [
            'Asymetric',
            'Self-regulating starting temperature'
        ]
        for param in checkbox_list:
            value = tk.BooleanVar(value=False)
            frame = tk.Frame(self.tab2, bg="white", pady=10)
            frame.pack(fill=tk.X)
            label = tk.Label(
                frame,
                text=param,
                font=("Arial", 14),
                bg="white",
                anchor="w",
                fg="black",
            )
            label.pack(side=tk.LEFT)
            value_checkbox = tk.Checkbutton(
                frame,
                variable=value,
                bg="white",
                fg="black",
            )
            value_checkbox.pack(padx=(0, 25), side=tk.RIGHT)
            self.param_entries_quangdai[param] = value
            
        # create apply button
        apply_button = tk.Button(
            self.tab2,
            text="Apply",
            font=("Arial", 14, "bold"),
            bg="purple",  # Light purple background
            fg="black",
            bd=0,
            padx=1,
            pady=1,
            command= self.run_simulation_quangdai
        )
        apply_button.pack(fill=tk.X, pady=(5, 0))
    def run_simulation_quangdai(self):
        """Run the simulated annealing algorithm and update the visualization"""
        
        # Get parameters from input fields
        dict_neightbor = {
            "swap within route": 0,
            "two opt within route": 1,
            "swap between routes": 2,
            "move customer": 3,
            "random": 4
        }

        random_seed = int(self.param_entries_quangdai["Random Seed"].get())

        num_customers = int(self.param_entries_quangdai["Number of Customers"].get())
        num_vehicles = int(self.param_entries_quangdai["Number of Vehicles"].get())
        initial_temp = float(self.param_entries_quangdai["Initial Temperature"].get())
        cooling_rate = float(self.param_entries_quangdai["Cooling rate"].get())
        max_iter = int(self.param_entries_quangdai["Maximum iteration"].get())

        neighbor_method = self.param_entries_quangdai["Neighbor Method"].get()
        neighbor_method = dict_neightbor[neighbor_method]

        initialization_method = self.param_entries_quangdai["Initialization Method"].get()
        cooling_strategy = self.param_entries_quangdai["Cooling Strategy"].get()
        asymetric = self.param_entries_quangdai["Asymetric"].get()
        self_regulating_starting_temp = self.param_entries_quangdai["Self-regulating starting temperature"].get()
        operator_mode = None

        if neighbor_method == 4:
            operator_mode = 'random'
        else:
            operator_mode = None

        quangdai.run_simulation(
            custom_customer_map = [],
            custom_customer_map_mode = False,
            custom_cost_matrix=[],
            custom_cost_matrix_mode=False,
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            asymmetric_percentage = 0.25,
            use_asymmetric=asymetric,
            max_factor = 2.0,
            auto_initial_temp = self_regulating_starting_temp,
            sample_size = 100,
            desired_accept_prob=0.8, 
            acceptable_error=0.01, 
            p=1.0,
            max_iter_temp = 100,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            stopping_temp=1e-50,
            max_iter=max_iter,
            depot=0,
            initialization_method=initialization_method,
            cooling_strategy=cooling_strategy,
            operator_mode=operator_mode,
            chosen_operator=neighbor_method,
            random_seed=random_seed
        )

        # Load the image and display it
        self.image = tk.PhotoImage(file="result_quangdai.png")

        # Destroy old frame if it exists
        if self.image_frame is not None:
            self.image_frame.destroy()
        
        # Image frame
        self.image_frame = tk.Frame(self.visual_frame, width=400, height=400, bg="white")
        self.image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        tk.Label(self.image_frame, image=self.image).pack(padx=0, pady=40)
        
    def run_simulation_giahy(self):
        """Run the simulated annealing algorithm and update the visualization"""
        
        # Get parameters from input fields
        dict_neightbor = {
            "two opt": giahy.two_opt,
            "three opt": giahy.three_opt,
            "random swap": giahy.random_swap
        }

        num_cities = int(self.param_entries["Numbers of Cities"].get())
        initial_temp = float(self.param_entries["Initial Temperature"].get())
        cooling_rate = float(self.param_entries["Cooling Rate"].get())
        max_iter = int(self.param_entries["Maximum Iteration"].get())
        stopping_temp = float(self.param_entries["Stopping Temperature"].get())

        neighbor_method = self.param_entries["Neighbor Method"].get()
        neighbor_method = dict_neightbor[neighbor_method]

        initialization_method = self.param_entries["Initialization Method"].get()
        cooling_strategy = self.param_entries["Cooling Strategy"].get()
        OnlyNearestNeighbor = self.param_entries["Only Nearest Neighbor"].get()
        CustomCity = self.list_of_coordinates

        # print(CustomCity)
        # print(neighbor_method)

        def real_distance(city1, city2, radius=6371.0):
            lat1, lon1 = map(math.radians, city1)
            lat2, lon2 = map(math.radians, city2)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return radius * c

        def total_distance_real(tour, cities):
            return sum(real_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))
        
        
        start_time = time.time() #Đếm thời gian chạy
        best_tour, best_cost, cities, iterations, neighbor_name, initialization_name = giahy.simulated_annealing(
            num_cities=num_cities,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            stopping_temp=stopping_temp,
            max_iter=max_iter,
            neighbor_method=neighbor_method,
            initialization_method=initialization_method,
            cooling_strategy=cooling_strategy,
            OnlyNearestNeighbor=OnlyNearestNeighbor,
            CustomCity=CustomCity
        )

        end_time = time.time()  #Đếm thời gian chạy

        execution_time = round(end_time - start_time, 2) 

        realdistance = total_distance_real(best_tour, cities)

        giahy.plot_tour(
            best_tour, cities, iterations, num_cities, 
            initial_temp, cooling_rate, stopping_temp, max_iter, neighbor_name, OnlyNearestNeighbor, cooling_strategy,
            best_cost, execution_time, initialization_name,realdistance, filename="result_giahy.png")

        # Load the image and display it
        self.image = tk.PhotoImage(file="result_giahy.png")

        # Destroy old frame if it exists
        if self.image_frame is not None:
            self.image_frame.destroy()
        
        # Image frame
        self.image_frame = tk.Frame(self.visual_frame, width=400, height=400, bg="white")
        self.image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        tk.Label(self.image_frame, image=self.image).pack(padx=0, pady=40)

        # Hiển thị bản đồ
        map_plot = giahy.plot_map(best_tour, cities)
        map_plot.save("tsp_map.html")  # Lưu ra file HTML


        # Lưu file
        file_path = "tsp_map.html"
        map_plot.save(file_path)

        # Lấy đường dẫn tuyệt đối và mở bằng trình duyệt mặc định
        full_path = os.path.abspath(file_path)
        webbrowser.open(f"file://{full_path}")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatedAnnealingDemo(root)
    root.mainloop()