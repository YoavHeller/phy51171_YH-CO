# analysis.py
import numpy as np
import matplotlib.pyplot as plt
from production import results,steps


class DataProcessor:
    def __init__(self, data):
        self.num_cells = len(data[0][0])
        self.q = data[0][2]
        self.m = data[0][3][0]
        self.electric_field = data[1]
        self.positions = data[3]
        self.velocities = data[4]
        self.time_steps = len(data[1])
        self.density_profiles = data[5]
        
        #  Data storage overtime
        self.energy_overtime = []
        self.momentum_overtime = []
        self.density_overtime = []

    def energy(self):
        for t in range(self.time_steps):
            kinetic_energy = 0
            for i in range(len(self.velocities[t])):
                kinetic_energy += self.m[i] * self.velocities[t][i] ** 2/2
            self.energy_overtime.append(kinetic_energy)
        return self.energy_overtime

    def momentum(self):
        for t in range(self.time_steps):
            moment = 0
            for i in range(len(self.velocities[t])):
                moment +=  self.m[i] * abs(self.velocities[t][i])
            self.momentum_overtime.append(moment)
        print(self.momentum_overtime)
        return self.momentum_overtime
    
    def analysis(self):
        energy = self.energy()
        momentum = self.momentum()
        return [self.density_profiles,energy,momentum]


data = results
num_steps = steps

processor = DataProcessor(data)
analysis = processor.analysis()
full_data_array = [data,analysis,num_steps]
print(full_data_array[1][0])
