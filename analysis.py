# analysis.py
import numpy as np
import matplotlib.pyplot as plt
from production import results,steps
import pandas as pd


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
        self.density_overtime = []

    def energy(self):
        for t in range(self.time_steps):
            kinetic_energy = 0
            for i in range(len(self.velocities[t])):
                kinetic_energy += self.m[i] * self.velocities[t][i] ** 2/2
            self.energy_overtime.append(kinetic_energy)
        return self.energy_overtime
    
    def analysis(self):
        energy = self.energy()
        return [self.density_profiles,energy]


data = results
num_steps = steps

processor = DataProcessor(data)
analysis = processor.analysis()
full_data_array = [data,analysis,num_steps]
print(full_data_array[1][0])

# Save results to Excel
output_data = {
    'Electric Field': processor.electric_field,
    'Positions': [list(pos) for pos in processor.positions],
    'Velocities': [list(vel) for vel in processor.velocities],
    'Density Profiles': processor.density_profiles,
    'Kinetic Energy': analysis[1],
    'Number of Steps': num_steps
}

df = pd.DataFrame(output_data)

# Specify the file path
file_path = "/Users/charlotte/Desktop/analysis_results.xlsx"
df.to_excel(file_path, index=False)

with pd.ExcelWriter(file_path) as writer:
    # Save Electric Field
    pd.DataFrame(output_data['Electric Field']).to_excel(writer, sheet_name="Electric Field", index=False)
    # Save Positions
    pd.DataFrame(output_data['Positions']).to_excel(writer, sheet_name="Positions", index=False)
    # Save Velocities
    pd.DataFrame(output_data['Velocities']).to_excel(writer, sheet_name="Velocities", index=False)
    # Save Density Profiles
    pd.DataFrame(output_data['Density Profiles']).to_excel(writer, sheet_name="Density Profiles", index=False)
    # Save Kinetic Energy
    pd.DataFrame({'Kinetic Energy': output_data['Kinetic Energy']}).to_excel(writer, sheet_name="Kinetic Energy", index=False)
    # Save Number of Steps
    pd.DataFrame({'Number of Steps': [output_data['Number of Steps']]}).to_excel(writer, sheet_name="Summary", index=False)

print(f"Data saved to {file_path}")
