import numpy as np
import matplotlib as plt


def plot_density_profile(simulation):
   density_profile = density_profile(self, simulation.num_cells)
   plt.figure(figsize=(10, 6))
   plt.scatter(density_profile, [t] * len(density_profile), label=f"Step {t}")
   plt.xlabel("density")
   plt.ylabel("Time Step")
   plt.title("Density Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
   plt.tight_layout()
   plt.show()
   
def plot_energy_vs_time(simulation):
   energy = energy(self,simulation.m, simulation.q)
   plt.figure(figsize=(10, 6))
   plt.scatter(energy[0], [t] * len(energy), label=f"Step {t}")
   plt.xlabel("kinetic and electric energy")
   plt.ylabel("Time Step")
   plt.title("Kinetic and electric energy Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
   plt.tight_layout()
   plt.show()

def plot_momentum(simulation):
   momentum = momentum(self, simulation.num_cells)
   plt.figure(figsize=(10, 6))
   plt.scatter(momentum, [t] * len(density_profile), label=f"Step {t}")
   plt.xlabel("momentum")
   plt.ylabel("Time Step")
   plt.title("Momemtum Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
   plt.tight_layout()
   plt.show()
   
def plot_field_profile(simulation):
   field = simulation.data[1]
   plt.figure(figsize=(10, 6))
   plt.scatter(Field, [t] * len(density_profile), label=f"Step {t}")
   plt.xlabel("momentum")
   plt.ylabel("Time Step")
   plt.title("Momemtum Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
   plt.tight_layout()
   plt.show()





   
