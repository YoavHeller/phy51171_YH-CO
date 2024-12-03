# analysis.py
import numpy as np
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, data):
        self.num_cells = len(data[0][0])
        self.q = data[0][3]
        self.m = data[0][4]
        self.electric_field = data[1]
        self.positions = data[3]
        self.velocities = data[4]
        self.time_steps = len(data[1])
        
        #  Data storage overtime
        self.energy = []
        self.momentum = []
        self.density_overtime = []

    def energy(self):
        for t in range(len(self.time_steps)):
            kinetic_energy = self.m * np.sum(self.velocities[t] ** 2)/2
            electric_energy = self.q * np.sum(self.electric_field[t] ** 2)/2
            self.energy.append((kinetic_energy, electric_energy))
        return self.energy

    def momentum(self):
        for t in range(len(self.time_steps)):
            momentum = self.m * np.sum(self.velocities[t])
            self.momentum.append(momentum)
        return momentum

    def density_profile(self):
        density_profiles = []
        for t in range(len(self.time_steps)):
            density, _ = np.histogram(self.positions[t], bins=self.num_cells, range=(0, self.num_cells+1))
            density_profiles.append(density)
        self.density_overtime = density_profiles
        return density_profiles
    
    def analysis(self):
        density_profile = density_profile(self)
        energy = energy(self)
        momentum = momentum(self)


