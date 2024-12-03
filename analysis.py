# analysis.py
import numpy as np
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, positions, velocities, electric_field, time_steps):
        self.positions = positions
        self.velocities = velocities
        self.electric_field = electric_field
        self.time_steps = time_steps
        
        #  Data storage overtime
        self.energy = []
        self.momentum = []
        self.density_overtime = []

    def energy(self, m, q):
        for t in range(len(self.time_steps)):
            kinetic_energy = m * np.sum(self.velocities[t] ** 2)/2
            electric_energy = q * np.sum(self.electric_field[t] ** 2)/2
            self.energy.append((kinetic_energy, electric_energy))
        return self.energy

    def momentum(self, m):
        for t in range(len(self.time_steps)):
            total_momentum = m * np.sum(self.velocities[t])
            self.momentum.append(total_momentum)
        return momentum

    def density_profile(self, num_cells):
        density_profiles = []
        for t in range(len(self.time_steps)):
            density, _ = np.histogram(self.positions[t], bins=num_cells, range=(0, grid_points))
            density_profiles.append(density)
        self.density_overtime = density_profiles
        return density_profiles
    
    def analysis(self,simulation,show_density,show_energy, show_momentum):
        if show_density:
            density_profile = density_profile(self, simulation.num_cells)
            plt.figure(figsize=(10, 6))
            plt.scatter(density_profile, [t] * len(density_profile), label=f"Step {t}")
            plt.xlabel("density")
            plt.ylabel("Time Step")
            plt.title("Density Over Time (20 Particles)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
            plt.tight_layout()
            plt.show()
        if show_energy:
            energy = energy(self,simulation.m, simulation.q)
            plt.figure(figsize=(10, 6))
            plt.scatter(energy, [t] * len(energy), label=f"Step {t}")
            plt.xlabel("energy")
            plt.ylabel("Time Step")
            plt.title("Energy Over Time (20 Particles)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
            plt.tight_layout()
            plt.show()
        if show_momentum:
            momentum = momentum(self, simulation.m)
            plt.figure(figsize=(10, 6))
            plt.scatter(momentum, [t] * len(density_profile), label=f"Step {t}")
            plt.xlabel("momentum")
            plt.ylabel("Time Step")
            plt.title("Momentum Over Time (20 Particles)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
            plt.tight_layout()
            plt.show()
        


