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


# ---- Run the Algorithm ----
#num_cities = int(input("Enter the number of cities: "))
num_cities = 20
best_tour, best_cost, cities = simulated_annealing(num_cities)


# ---- Print the Result ----
print("Best tour:", best_tour)
print("Best cost:", round(best_cost, 2))

# ---- Optional: Plot the Result ----
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

# Call the function and save to file
plot_tour(best_tour, cities, filename="result.png")