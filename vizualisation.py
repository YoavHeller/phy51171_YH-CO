import numpy as np
import matplotlib as plt


def plot_density_profile():
   density_profile = DataProcessor.analysis.density_profile()
   plt.figure(figsize=(10, 6))
   plt.plot([t for t in range(analysis.time_steps)],density_profile)
   plt.xlabel("Time Step")
   plt.ylabel("Density")
   plt.title("Density Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()
   
def plot_energy_vs_time():
   energy = DataProcessor.analysis.energy()
   plt.figure(figsize=(10, 6))
   plt.plot([t for t in range(DataProcessor.analysis.time_steps)],energy)
   plt.ylabel("Kinetic energy")
   plt.xlabel("Time Step")
   plt.title("Kinetic energy Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()

def plot_momentum():
   momentum = DataProcessor.analysis.momentum()
   plt.figure(figsize=(10, 6))
   plt.plot([t for t in range(DataProcessor.analysis.time_steps)],momentum)
   plt.ylabel("Momentum")
   plt.xlabel("Time Step")
   plt.title("Momemtum Over Time")
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()
   
def plot_field_profile(time):
   field = DataProcessor.field()[time]
   plt.figure(figsize=(10, 6))
   plt.plot([x for x in range(DataProcessor.analysis.num_cells)], field)
   plt.xlabel("Position")
   plt.ylabel("Field")
   plt.title("Field at time" + time)
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
   plt.tight_layout()
   plt.show()





   
