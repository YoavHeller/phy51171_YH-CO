import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import analysis
from analysis import full_data_array


# data de la forme [data, density_profiles,energy,momentum]


def plot_density_profile(full_data_array):
    density_profile = full_data_array[5]
    grid_pos = full_data_array[0][0]
    Xi = grid_pos[0]
    Xf = grid_pos[-1]
    fig, ax = plt.subplots()
    ax.set_xlim(Xi, Xf)
    plt.figure(figsize=(10, 6))
    plt.plot([t for t in range(len(full_data_array[1]))], density_profile)
    plt.xlabel("Time Step")
    plt.ylabel("Density")
    plt.title("Density Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_energy_vs_time(full_data_array):
    data = full_data_array[0]
    energy = full_data_array[1][1]
    grid_pos = data[0][0]
    Xi = grid_pos[0]
    Xf = grid_pos[len(grid_pos) - 1]
    fig, ax = plt.subplots()
    ax.set_xlim(Xi, Xf)
    plt.plot([t for t in range(analysis.DataProcessor.time_steps)], energy)
    plt.ylabel("Kinetic energy")
    plt.xlabel("Time Step")
    plt.title("Kinetic energy Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_momentum(full_data_array, step):
    data = full_data_array[0]
    momentum = full_data_array[1][2]
    grid_pos = data[0][0]
    Xi = grid_pos[0]
    Xf = grid_pos[len(grid_pos) - 1]
    fig, ax = plt.subplots()
    ax.set_xlim(Xi, Xf)
    plt.plot([t for t in range(step)], momentum)
    plt.ylabel("Momentum")
    plt.xlabel("Time Step")
    plt.title("Momemtum Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_field_profile(full_data_array, step):
    data = full_data_array[0]
    field = data[1]
    positions = data[3]
    grid_pos = data[0][0]
    Xi = grid_pos[0]
    Xf = grid_pos[len(grid_pos) - 1]
    fig, ax = plt.subplots()
    ax.set_xlim(Xi, Xf)
    ax.scatter(grid_pos, field[step], color='b', marker='o', s=1, linewidths=0.5)
    plt.xlabel("Position")
    plt.ylabel("Field")
    plt.title("Field at step " + str(step))
    plt.tight_layout()
    plt.show()


def plot_velocity_profile(full_data_array, step):
    velocities = full_data_array[4]
    positions = full_data_array[3]
    grid_pos = full_data_array[0][0]
    Xi = grid_pos[0]
    Xf = grid_pos[-1]
    fig, ax = plt.subplots()
    ax.set_xlim(Xi, Xf)
    plt.scatter(positions[step][:len(positions[step])//2], velocities[step][:len(velocities[step])//2])
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Velocity at step " + str(step))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


print(plot_velocity_profile(full_data_array, 10))
#print(plot_field_profile(full_data_array, 10))
#print(plot_momentum(full_data_array))
#print(plot_energy_vs_time(full_data_array))
#plot_density_profile(full_data_array)
