import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import analysis
from analysis import full_data_array


# data de la forme [data, density_profiles,energy,momentum]

def plot_density_profile(full_data_array,step):
   data = full_data_array[0]
   positions = data[3]
   density_profiles = full_data_array[1][0]
   grid_pos = data[0][0]
   Xi = grid_pos[0]
   Xf = grid_pos[len(grid_pos)-1]
   fig, ax = plt.subplots()
   ax.set_xlim(Xi,Xf)
   plt.figure(figsize=(10, 6))
   plt.bar(grid_pos, density_profiles[step], width=(Xf - Xi) / len(grid_pos), align='center', alpha=0.7, edgecolor="k")
   plt.ylabel("Density")
   plt.title("Density at step "+ str(step))
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()
 
def plot_energy_vs_time(full_data_array):
   data = full_data_array[0]
   energy = full_data_array[1][1]
   num_steps = full_data_array[2]
   grid_pos = data[0][0]
   fig, ax = plt.subplots()
   ax.set_xlim(0,num_steps)
   plt.plot([t for t in range(num_steps+1)],energy)
   plt.ylabel("Kinetic energy")
   plt.xlabel("Time Step")
   plt.title("Kinetic energy Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()

def plot_momentum(full_data_array):
   data = full_data_array[0]
   num_steps = full_data_array[2]
   momentum = full_data_array[1][2]
   grid_pos = data[0][0]
   fig, ax = plt.subplots()
   ax.set_xlim(0,num_steps)
   plt.plot([t for t in range(num_steps+1)],momentum)
   plt.ylabel("Momentum")
   plt.xlabel("Time Step")
   plt.title("Momemtum Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()
   
def plot_field_profile(full_data_array,step):
   data = full_data_array[0]
   field = data[1]
   positions = data[3]
   grid_pos = data[0][0]
   Xi = grid_pos[0]
   Xf = grid_pos[len(grid_pos)-1]
   fig, ax = plt.subplots()
   ax.set_xlim(Xi,Xf)
   ax.scatter(grid_pos, field[step], color='b', marker='o',s=1, linewidths=0.5)
   plt.xlabel("Position")
   plt.ylabel("Field")
   plt.title("Field at step " + str(step))
   plt.tight_layout()
   plt.show()

def plot_velocity_profile(full_data_array,step):
   data = full_data_array[0]
   velocities = data[4]
   positions = data[3]
   grid_pos = data[0][0]
   Xi = grid_pos[0]
   Xf = grid_pos[len(grid_pos)-1]
   fig, ax = plt.subplots()
   ax.set_xlim(Xi,Xf)
   ax.scatter(positions[step][:len(positions[step])//2], velocities[step][:len(velocities[step])//2])
   plt.xlabel("Position")
   plt.ylabel("Velocity")
   plt.title("Velocity at step "+ str(step))
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()


print(plot_velocity_profile(full_data_array,10))
print(plot_field_profile(full_data_array,10))
print(plot_density_profile(full_data_array,4))
print(plot_momentum(full_data_array))
print(plot_energy_vs_time(full_data_array))
