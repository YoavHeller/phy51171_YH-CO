   def analysis(self,simulation,show_density,show_energy, show_momentum):
        if show_density:
            density_profile = density_profile(self, simulation.num_cells)
            plt.figure(figsize=(10, 6))
            plt.scatter(density_profile, [t] * len(density_profile), label=f"Step {t}")
            plt.xlabel("density")
            plt.ylabel("Time Step")
            plt.title("Density Over Time")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
            plt.tight_layout()
            plt.show()
        if show_energy:
            energy = energy(self,simulation.m, simulation.q)
            plt.figure(figsize=(10, 6))
            plt.scatter(energy, [t] * len(energy), label=f"Step {t}")
            plt.xlabel("energy")
            plt.ylabel("Time Step")
            plt.title("Energy Over Time")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
            plt.tight_layout()
            plt.show()
        if show_momentum:
            momentum = momentum(self, simulation.m)
            plt.figure(figsize=(10, 6))
            plt.scatter(momentum, [t] * len(momentum), label=f"Step {t}")
            plt.xlabel("momentum")
            plt.ylabel("Time Step")
            plt.title("Momentum Over Time")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
            plt.tight_layout()
            plt.show()