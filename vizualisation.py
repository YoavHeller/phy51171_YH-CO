import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import analysis
from analysis import full_data_array


# data de la forme [data, density_profiles,energy,momentum]


def plot_density_profile(full_data_array):
   density_profile = full_data_array[1]
   plt.figure(figsize=(10, 6))
   plt.plot([t for t in range(analysis.DataProcessor.time_steps)],density_profile)
   plt.xlabel("Time Step")
   plt.ylabel("Density")
   plt.title("Density Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()
   
def plot_energy_vs_time(full_data_array):
   energy = full_data_array[2]
   plt.figure(figsize=(10, 6))
   plt.plot([t for t in range(analysis.DataProcessor.time_steps)],energy)
   plt.ylabel("Kinetic energy")
   plt.xlabel("Time Step")
   plt.title("Kinetic energy Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()

def plot_momentum(full_data_array):
   momentum = full_data_array[3]
   plt.figure(figsize=(10, 6))
   plt.plot([t for t in range(analysis.DataProcessor.time_steps)],momentum)
   plt.ylabel("Momentum")
   plt.xlabel("Time Step")
   plt.title("Momemtum Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()
   
def plot_field_profile(full_data_array,step):
   data = full_data_array[0]
   field = data[1]
   num_cells = data[0][0]
   plt.figure(figsize=(10, 6))
   plt.plot([x for x in range(num_cells)], field[step])
   plt.xlabel("Position")
   plt.ylabel("Field")
   plt.title("Field at step" + step)
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()

def plot_velocity_profile(full_data_array,step):
   data = full_data_array[0]
   velocities = data[4]
   num_cells = len(data[0][0])
   plt.figure(figsize=(10, 6))
   plt.plot([x for x in range(num_cells)], velocities[step])
   plt.xlabel("Position")
   plt.ylabel("Velocity")
   plt.title("Velocity at step" + step)
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()


print(plot_velocity_profile(full_data_array,2))
