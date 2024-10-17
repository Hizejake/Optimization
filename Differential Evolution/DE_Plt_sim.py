import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, ImageMagickWriter
def rastrigin(X):
    if isinstance(X[0], (int, float)):
        X = [[X[i]] for i in range(len(X))]
    
    val = []
    for xi in X:
        fx = 10 * len(xi) + sum(np.array(xi) ** 2 - 10 * np.cos(2 * np.pi * np.array(xi)))
        val.append(fx)
    return np.array(val)

def generate_target(num_variables, population_size, x_min, x_max):
    return np.random.uniform(x_min, x_max, (population_size, num_variables))

def diff_evol(num_iterations, population_size, num_variables, x_min, x_max, scale_factor, crossover_probability):
    target = generate_target(num_variables, population_size, x_min, x_max)
    trial = np.zeros_like(target)
    
    populations = [target.copy()]
    
    for i in range(num_iterations):
        mutant = np.clip(np.array([random.sample(list(target), 3)[0] + scale_factor * (random.sample(list(target), 3)[1] - (random.sample(list(target), 3)[2])) for _ in range(population_size)]), x_min, x_max)
        
        for j in range(population_size):  
            I_rand = np.random.randint(0, num_variables)
            for k in range(num_variables):  
                if np.random.uniform(0, 1) <= crossover_probability or k == I_rand:
                    trial[j, k] = mutant[j, k]
                else:
                    trial[j, k] = target[j, k]
        
        target_dict = {tuple(target[i]): rastrigin(target)[i] for i in range(population_size)}
        trial_dict = {tuple(trial[i]): rastrigin(trial)[i] for i in range(population_size)}
        
        target_dict.update(trial_dict)
        items = target_dict.items()
        items = sorted(items, key=lambda x: x[1])
        new_target = [np.array(item[0]) for item in items[:population_size]]
        
        target = np.array(new_target)
        populations.append(target.copy())
    
    return target[0], populations

# Run the differential evolution algorithm and get the populations
best_solution, populations = diff_evol(50, 50, 2, -5.12, 5.12, 0.5, 0.9)

# Generate Rastrigin function values for the contour plot
x = np.linspace(-5.12, 5.12, 400)
y = np.linspace(-5.12, 5.12, 400)
X, Y = np.meshgrid(x, y)
Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

# Create contour plot with increased resolution
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
contour = ax.contour(X, Y, Z, levels=50, cmap='viridis')
fig.colorbar(contour)

# Initialize scatter plot for the population
scat = ax.scatter([], [], s=50, color='red')
generation_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', fontsize=12, color='black')

def init():
    ax.set_xlim(-5.12, 5.12)
    ax.set_ylim(-5.12, 5.12)
    generation_text.set_text('')
    return scat, generation_text

def update(frame):
    scat.set_offsets(populations[frame])
    generation_text.set_text(f'Generation: {frame}')
    return scat, generation_text

ani = animation.FuncAnimation(fig, update, frames=len(populations), init_func=init, blit=True)
ani.save('differential_evolution.gif', writer= animation.PillowWriter(fps=10))
plt.show()