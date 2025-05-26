import tkinter as tk
from tkinter import ttk

import math
import random
import matplotlib.pyplot as plt


# # ---- Generate Random Cities ----
# def generate_cities(num_cities):
#     return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_cities)]
# #random 2 số thập phân trong khoảng 0 - 10 để làm tọa độ x, y cho các thành phố


# # ---- Distance Calculation ----
# def distance(city1, city2):
#     return math.hypot(city1[0] - city2[0], city1[1] - city2[1])
# #tính độ dài cạnh huyền dựa trên x và y (x và y là cạng vuông dọc theo trục tọa độ)


# def total_distance(tour, cities):
#     return sum(distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))
# # khi i chạy tới tour cuối cùng, ví dụ 7 thì 7 % 7 = 0 quay về city đầu, nếu ko thì 2 % 7 = 2, 3 % 7= 3 ,...


# # ---- 2-opt Swap ----
# def two_opt(tour):
#     a, b = sorted(random.sample(range(len(tour)), 2))
#     new_tour = tour[:a] + tour[a:b+1][::-1] + tour[b+1:]
#     return new_tour


# # ---- Simulated Annealing Algorithm ----
# def simulated_annealing(num_cities, initial_temp=1000, cooling_rate=0.995, stopping_temp=1e-8, max_iter=100000):
#     cities = generate_cities(num_cities)
#     n = len(cities)
#     current_tour = list(range(n)) #list = 0,1,2,.. = số thành phố
#     random.shuffle(current_tour) #trộn list
#     current_cost = total_distance(current_tour, cities)
#     best_tour = current_tour[:]
#     best_cost = current_cost
#     temp = initial_temp


#     for i in range(max_iter):
#         if temp < stopping_temp:
#             break


#         new_tour = two_opt(current_tour)
#         new_cost = total_distance(new_tour, cities)
#         delta = new_cost - current_cost


#         if delta < 0 or random.random() < math.exp(-delta / temp):
#             current_tour = new_tour
#             current_cost = new_cost
#             if current_cost < best_cost:
#                 best_tour = current_tour[:]
#                 best_cost = current_cost


#         temp *= cooling_rate


#     return best_tour, best_cost, cities


# # ---- Run the Algorithm ----
# #num_cities = int(input("Enter the number of cities: "))
# num_cities = 20
# best_tour, best_cost, cities = simulated_annealing(num_cities)


# # ---- Print the Result ----
# print("Best tour:", best_tour)
# print("Best cost:", round(best_cost, 2))

# # ---- Optional: Plot the Result ----
# def plot_tour(tour, cities):
#     tour_cities = [cities[i] for i in tour] + [cities[tour[0]]]
#     xs, ys = zip(*tour_cities)
#     plt.figure(figsize=(10, 6))
#     plt.plot(xs, ys, marker='o')
#     for idx, (x, y) in enumerate(cities):
#         plt.text(x, y, str(idx), fontsize=12, ha='right', va='bottom')
#     plt.title('TSP Tour (Simulated Annealing + 2-opt)')
#     plt.grid(True)
#     plt.show()


# plot_tour(best_tour, cities)


# root = tk.Tk()
# root.title("Widgets Demo")

# # popular labels
# widgets = [
#     tk.Label,
#     tk.Checkbutton,
#     ttk.Combobox,
#     tk.Entry,
#     tk.Button,
#     tk.Radiobutton,
#     tk.Scale,
#     tk.Spinbox,
# ]

# for widget in widgets:
#     try:
#         widget = widget(root, text=widget.__name__)
#     except tk.TclError:
#         widget = widget(root)
#     widget.pack(padx=5, pady=5, fill="x")

# root.mainloop()
#================

# import tkinter as tk

# root = tk.Tk()
# root.title("Tkinter Scale")
# root.geometry("200x80")

# def value_changed(event):
#     label.config(text=event.widget.get())

# scale = tk.Scale(root, from_=0, to=10, orient="horizontal")
# scale.bind( value_changed)
# scale.pack(padx=5, pady=5, fill="x")

# # A helper label to show the selected value
# label = tk.Label(root, text="0")
# label.pack(padx=5, pady=5, fill="x")

# root.mainloop()

#================#================#================

# import tkinter as tk

# root = tk.Tk()
# root.title("Frame Demo")
# root.config(bg="skyblue")

# # Create Frame widget
# frame = tk.Frame(root, width=200, height=200)
# frame.pack(padx=10, pady=10)

# nested_frame = tk.Frame(frame, width=200, height=200, bg="red")
# nested_frame.pack(padx=10, pady=10)

# root.mainloop()

#================#================#================

import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Image Editor")

image = tk.PhotoImage(file = "result.png")

# # Tools frame
# tools_frame = tk.Frame(root, width=200, height=400, bg="skyblue")
# tools_frame.pack(padx=5, pady=5, side=tk.LEFT, fill=tk.Y)
# tk.Label(
#     tools_frame,
#     text="Original Image",
#     bg="skyblue",
# ).pack(padx=5, pady=5)
# thumbnail_image = image.subsample(5, 5)
# tk.Label(tools_frame, image=thumbnail_image).pack(padx=5, pady=5)

# # Tools and Filters tabs
# notebook = ttk.Notebook(tools_frame)
# notebook.pack(expand=True, fill="both")

# tools_tab = tk.Frame(notebook, bg="lightblue")
# tools_tab.pack(padx=5, pady=5)
# tools_var = tk.StringVar(value="None")
# for tool in ["Resizing", "Rotating"]:
#     tk.Radiobutton(
#         tools_tab,
#         text=tool,
#         variable=tools_var,
#         value=tool,
#         bg="lightblue",
#     ).pack(anchor="w", padx=20, pady=5)

# filters_tab = tk.Frame(notebook, bg="lightgreen")
# filters_var = tk.StringVar(value="None")
# for filter in ["Blurring", "Sharpening"]:
#     tk.Radiobutton(
#         filters_tab,
#         text=filter,
#         variable=filters_var,
#         value=filter,
#         bg="lightgreen",
#     ).pack(anchor="w", padx=20, pady=5)

# notebook.add(tools_tab, text="Tools")
# notebook.add(filters_tab, text="Filters")

# num_cities = 20
# best_tour, best_cost, cities = simulated_annealing(num_cities)


# Image frame
image_frame = tk.Frame(root, width=400, height=400, bg="grey")
image_frame.pack(padx=5, pady=5, side=tk.RIGHT)
tk.Label(
    image_frame,
    text="Edited Image",
    bg="grey",
    fg="white",
).pack(padx=5, pady=5)
tk.Label(image_frame, image=image).pack(padx=5, pady=5)

root.mainloop()




