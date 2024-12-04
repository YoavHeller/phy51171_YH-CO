# analysis.py
import numpy as np
import matplotlib.pyplot as plt
from production import results


class DataProcessor:
    def __init__(self, data):
        self.num_cells = len(data[0][0])
        self.positions=data[3]
        self.velocities=data[4]
        self.q = data[0][2]
        self.m = data[0][3]
        self.electric_field = data[1]
        self.time_steps = len(data[1])

        #  Data storage overtime
        self.energy_overtime = []
        self.momentum_overtime = []
        self.density_overtime = []

    def energy(self):
        for t in range(self.time_steps):
            kinetic_energy = self.m * np.sum(self.velocities[t] ** 2) / 2
            self.energy_overtime.append(np.array(kinetic_energy))
        return self.energy_overtime

    def momentum(self):
        for t in range(self.time_steps):
            moment = self.m * np.sum(self.velocities[t])
            self.momentum_overtime.append(moment)
        return self.momentum_overtime

    def density_profile(self):
        density_profiles = []
        for t in range(len(self.positions)):
            density, _ = np.histogram(self.positions[t], bins=self.num_cells, range=(0, self.num_cells + 1))
            density_profiles.append([np.array(density)])
        self.density_overtime = density_profiles
        return density_profiles

    def analysis(self):
        #density_profiles = self.density_profile() doesnt work not coded well
        energy = self.energy()
        momentum = self.momentum()
        return np.array([ energy, momentum], dtype=object)


data = results

processor = DataProcessor(data)
analysis = processor.analysis()
full_data_array = np.append(data, analysis)

print()
