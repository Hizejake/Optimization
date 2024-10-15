from manim import *
import numpy as np

def pso1(num_iterations, swarm_size, num_variables, x_min, x_max, alpha, beta, gamma, epsilon):
    def rastrigin(X):
        return 10 * len(X) + sum([(x ** 2 - 10 * np.cos(2 * np.pi * x)) for x in X])

    def generate_particle(num_variables, swarm_size, x_min, x_max):
        swarm = np.random.uniform(x_min, x_max, (swarm_size, num_variables))
        velocity = np.random.uniform(-1, 1, (swarm_size, num_variables))
        return swarm, velocity

    swarm, velocity = generate_particle(num_variables, swarm_size, x_min, x_max)
    swarm_positions = [swarm.tolist()]

    for i in range(num_iterations):
        swarm2 = swarm + epsilon * velocity
        swarm2 = np.clip(swarm2, x_min, x_max)
        
        dicx = {tuple(swarm[i]): rastrigin(swarm[i]) for i in range(swarm_size)}
        dicy = {tuple(swarm2[i]): rastrigin(swarm2[i]) for i in range(swarm_size)}

        if i == 0:
            local_best = swarm
            global_best = list(min(dicx, key=dicx.get))
        else:
            for j in range(swarm_size):
                key1, value1 = list(dicx.items())[j]
                key2, value2 = list(dicy.items())[j]
                
                local_best[j] = list(key2) if value2 < value1 else list(key1)
                
                global_best = list(min(dicy, key=dicy.get)) if min(dicy.values()) < min(dicx.values()) else list(min(dicx, key=dicx.get))

        velocity = (alpha * velocity + np.random.uniform(0, beta) * (np.array(local_best) - np.array(swarm)) + np.random.uniform(0, gamma) * (np.full_like(swarm, global_best) - np.array(swarm)))
        
        swarm = swarm2
        swarm_positions.append(swarm.tolist())

    return global_best, swarm_positions

class PSOSimulation2D(Scene):
    def construct(self):
        num_iterations = 100
        swarm_size = 30
        num_variables = 2
        x_min = -5.12
        x_max = 5.12
        alpha = 0.5
        beta = 0.3
        gamma = 0.9
        epsilon = 0.1

        global_best, swarm_positions = pso1(num_iterations, swarm_size, num_variables, x_min, x_max, alpha, beta, gamma, epsilon)

        # Create axes
        axes = Axes(
            x_range=[x_min, x_max, 1],
            y_range=[x_min, x_max, 1],
            axis_config={"color": BLUE}
        )
        self.add(axes)

        # Create Rastrigin contour map
        def rastrigin_surface(x, y):
            return 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = np.linspace(x_min, x_max, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = rastrigin_surface(X, Y)

        contour = axes.plot_contour(
            lambda x, y: rastrigin_surface(x, y),
            x_range=[x_min, x_max],
            y_range=[x_min, x_max],
            contour_config={"stroke_color": BLUE, "stroke_width": 1}
        )
        self.add(contour)

        # Create particles
        particles = VGroup(*[
            Dot(point=axes.c2p(pos[0], pos[1]), color=YELLOW)
            for pos in swarm_positions[0]
        ])
        self.add(particles)

        # Animate particles
        for i in range(1, num_iterations):
            new_positions = swarm_positions[i]
            self.play(
                *[particle.animate.move_to(axes.c2p(pos[0], pos[1])) for particle, pos in zip(particles, new_positions)],
                run_time=0.1
            )

if __name__ == "__main__":
    from manim import *
    scene = PSOSimulation2D()
    scene.render()
