# analysis.py
import numpy as np
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, data):
        self.num_cells = data[0][0]
        self.q = data[0][2]
        self.m = data[0][3]
        self.electric_field = data[1]
        self.positions = data[3]
        self.velocities = data[4]
        self.time_steps = len(data[1])
        
        #  Data storage overtime
        self.energy_overtime = []
        self.momentum_overtime = []
        self.density_overtime = []

    def energy(self):
        for t in range(self.time_steps):
            kinetic_energy = self.m * np.sum(self.velocities[t] ** 2)/2
            self.energy_overtime.append(kinetic_energy)
        return self.energy_overtime

    def momentum(self):
        for t in range(self.time_steps):
            moment = self.m * np.sum(self.velocities[t])
            self.momentum_overtime.append(moment)
        return self.momentum_overtime

    def density_profile(self):
        density_profiles = []
        for t in range(self.time_steps):
            density, _ = np.histogram(self.positions[t], bins=self.num_cells, range=(0, self.num_cells+1))
            density_profiles.append(density)
        self.density_overtime = density_profiles
        return density_profiles
    
    def analysis(self):
        density_profiles = self.density_profile()
        energy = self.energy()
        momentum = self.momentum()
        return [density_profiles,energy,momentum]


data = [[2,1,2,3],[1,2,2],[1,2,2],[1,2,2],[1,2,2]]

processor = DataProcessor(data)
analysis = processor.analysis()
array = [data,analysis]

print(array)
