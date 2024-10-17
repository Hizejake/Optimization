import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from matplotlib.animation import FuncAnimation, ImageMagickWriter

def rastrigin(X):
    if isinstance(X[0], (int, float)):
        X = [[X[i]] for i in range(len(X))]
    
    val = []
    for xi in X:
        fx = 10 * len(xi) + sum(np.array(xi) ** 2 - 10 * np.cos(2 * np.pi * np.array(xi)))
        val.append(fx)
    return np.array(val)

def generate_particle(num_variables, swarm_size, x_min, x_max):
    swarm = np.random.uniform(x_min, x_max, (swarm_size, num_variables))
    velocity = np.zeros_like(swarm)
    return swarm, velocity

def pso1(num_iterations, swarm_size, num_variables, x_min, x_max, alpha, beta, gamma, epsilon):
    swarm, velocity = generate_particle(num_variables, swarm_size, x_min, x_max)
    swarm_positions = [swarm.tolist()]

    for i in range(num_iterations):
        swarm2 = swarm + epsilon * velocity
        swarm2 = np.clip(swarm2, x_min, x_max)
        
        dicx = {tuple(swarm[i]): rastrigin(swarm)[i] for i in range(swarm_size)}
        dicy = {tuple(swarm2[i]): rastrigin(swarm2)[i] for i in range(swarm_size)}

        if i == 0:
            local_best = swarm
            global_best = list(min(dicx, key=dicx.get))
        else:
            for j in range(swarm_size):
                key1, value1 = list(dicx.items())[j]
                key2, value2 = list(dicy.items())[j]
                
                local_best[j] = list(key2) if value2 < value1 else list(key1)
                
                # global_best = list(min(dicy, key=dicy.get)) if min(dicy.values()) < min(dicx.values()) else list(min(dicx, key=dicx.get))
                global_best = list(min(dicy, key=lambda k: dicy[k])) if np.argmin(dicy) < np.argmin(dicx) else list(min(dicx, key=lambda k: dicx[k]))

        velocity = (alpha * velocity + np.random.uniform(0, beta) * (np.array(local_best) - np.array(swarm)) + np.random.uniform(0, gamma) * (np.full_like(swarm, global_best) - np.array(swarm)))
        swarm = swarm2
        swarm_positions.append(swarm.tolist())
        
    return global_best, swarm_positions

# Parameters
num_iterations = 50
swarm_size = 50
num_variables = 2
x_min = -5.12
x_max = 5.12
alpha = 0.5
beta = 0.3
gamma = 0.9
epsilon = 0.1

# Run PSO
global_best, swarm_positions = pso1(num_iterations, swarm_size, num_variables, x_min, x_max, alpha, beta, gamma, epsilon)

# Generate Rastrigin function values for the contour plot
x = np.linspace(x_min, x_max, 400)
y = np.linspace(x_min, x_max, 400)
X, Y = np.meshgrid(x, y)
Z = 10 * num_variables + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

# Create contour plot
fig, ax = plt.subplots(figsize=(10, 8))  # Set higher resolution
contour = ax.contour(X, Y, Z, levels=50, cmap='viridis')
fig.colorbar(contour)

# Initialize particles
particles, = ax.plot([], [], 'ro')
generation_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    particles.set_data([], [])
    generation_text.set_text('')
    return particles, generation_text

def update(frame):
    positions = swarm_positions[frame]
    particles.set_data([pos[0] for pos in positions], [pos[1] for pos in positions])
    generation_text.set_text(f'Generation: {frame}')
    return particles, generation_text

ani = FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=True, repeat=False)

# Save the animation as a video with higher resolution
# ani.save('pso_animation.mp4', writer='ffmpeg', fps=10, dpi=200)
ani.save('pso_animation.gif', writer=ImageMagickWriter(fps=10))

plt.show()
