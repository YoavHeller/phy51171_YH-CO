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
        return self.momentum

    def density_profile(self, grid_points):
        density_profiles = []
        for t in range(len(self.time_steps)):
            density, _ = np.histogram(self.positions[t], bins=grid_points, range=(0, grid_points))
            density_profiles.append(density)
        self.density_overtime = density_profiles
        return density_profiles
    


