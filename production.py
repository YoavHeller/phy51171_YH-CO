import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from numba import jit
from scipy.integrate import quad


class particle:
    ## class of a particle

    def __init__(self, x, v, q, m):
        ## recives values for location (x), velocity (v), charge(q), and mass (m)
        self.x = x
        self.v = v
        self.q = q
        self.m = m

    def __iter__(self):
        return iter((self.x, self.v, self.q, self.m))

    def update_pos(self, dt):
        ## moves the particle with it's current velocity for the time jump dt
        ## x_new = x + v*dt
        self.x += self.v * dt

    def update_vel(self, Field, dt):
        ## changes the velocity by a given field, for time jump dt
        ## v_new = v+ a * dt = v+E*q/m * dt
        acceleration = (Field * self.q) / self.m
        self.v += acceleration * dt

    def applied_field(self,field):
        ##lineary calculates the field on the particle
        grid_pos, field_val, dx= field
        if self.x > grid_pos[-1] or self.x< grid_pos[0]:
            return 0.0
        nearest_grid_point = int(np.round(self.x / dx))
        field_applied=((self.x-grid_pos[nearest_grid_point])/dx)*field_val[nearest_grid_point]+((grid_pos[nearest_grid_point+1]-self.x)/dx)*field_val[nearest_grid_point+1]
        return field_applied



class field:
    ##class of the field
    def __init__(self, Xi, Xf, num_cells):
        ## input of borders of the grid xi xf, and number of the cells.
        ## parameters: dx cell size
        ## field: for the field (not used yet)
        ## grid_pos: list of grid point positions
        ## num celles: number of cells
        self.dx = (Xf - Xi) / num_cells
        self.grid_pos = np.linspace(Xi, Xf, num_cells)
        self.num_cells = num_cells

        self.field = np.zeros(num_cells)  # used?
        self.freqs = 2 * np.pi * np.array(np.linspace(-self.num_cells / 2, self.num_cells / 2 + 1, self.num_cells)) / (
                    self.dx * self.num_cells)
        self.field_FFT = np.zeros(num_cells)

    def __iter__(self):
        ## iter fun of field only returns the grid positions, and dx)
        return iter((self.grid_pos, self.field, self.dx))

    def compute_density_first_order_method(self, particles):
        ##Calculates charge density using 1st order method, taken to be the size of the grid cell
        ## returns the density as an array in the grids positions
        density = np.zeros(self.num_cells)
        for par in particles:
            x, v, q, m = par.x, par.v, par.q, par.m
            # Find the nearest grid point
            nearest_grid_point = int(np.round(x / self.dx))
            if nearest_grid_point < 0 or nearest_grid_point >= self.num_cells-1:
                continue
            Xi = self.grid_pos[nearest_grid_point]
            density[nearest_grid_point] += (q / self.dx) * (x - Xi + self.dx)
            Xiplusone = self.grid_pos[nearest_grid_point + 1]
            density[nearest_grid_point + 1] += (q / self.dx) * (Xiplusone - x)

        return density

    def compute_density_triangle(self, particles):
        ##Calculates particles density using triangle method, taken to be the size of two times the grid cell
        density = np.zeros(self.num_cells)
        for par in particles:
            (x, v, q, m) = par
            nearest_grid_point = int(np.round(x / self.dx))
            width = 2 * self.dx

            def squar_func(x):

                return lambda y: (abs(y - x) <= (width / 2)) * q / width

            # Define the range of cells to influence (±2 cells around nearest point)
            cellleft = max(int(np.floor((x - width / 2) / self.dx)), 0)
            cellright = min(int(np.floor((x + width / 2) / self.dx)), self.num_cells - 1)
            for cell in range(cellleft, cellright + 1):
                distance = abs(grid_pos[cell] - x)
                if distance <= width:
                    Ai, _ = quad(squar_func(x), self.grid_pos(cell), self.grid_pos(cell + 1))
                    density[cell] += Ai

        return density

    def compute_density_gaussian(self, particles):
        density = np.zeros(self.num_cells)
        sigma = 2 * self.dx

        def gaussian_func(x0):
            # Normalization factor for Gaussian distribution
            A = q / (np.sqrt(2 * np.pi) * sigma)
            return lambda x: A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

        for par in particles:
            (x, v, q, m) = par
            nearest_grid_point = int(np.round(x / dx))
            # Define bounds for integration based on Gaussian spread
            num_points_to_cover = int(3 * sigma / self.dx)  # Cover 3 sigma range for significant spread
            xleft = max(nearest_grid_point - num_points_to_cover, 0)
            xright = min(nearest_grid_point + num_points_to_cover, self.num_cells - 1)
            for cell in range(xleft, xright + 1):
                Ai, _ = quad(gaussian_func(x), self.grid_pos[cell], self.grid_pos[cell + 1])
                density[cell] += Ai

        return density

    def density_field(self, density):
        ## finds the electric potential and electrical field of the density on the grid
        ## puts the values in field and field_FFT
        epsilon = 8.85419 * 10 ** (-12)  ##Units of ??

        density_FFT = np.fft.fft(density)

        a=self.freqs*self.dx

        potential_correction=np.sinc(a/(2*np.pi))
        potential_correction=np.nan_to_num(potential_correction, nan=0.0)
        potential_correction = self.freqs * potential_correction

        potential_FFT = density_FFT / (epsilon * (potential_correction ** 2))

        field_correction=np.sinc(a/(np.pi))
        field_correction=np.nan_to_num(field_correction, nan=0.0)
        field_correction = self.freqs* field_correction

        field_FFT_Sol = -1j * field_correction * potential_FFT
        field_Sol = np.fft.ifft(field_FFT_Sol)

        self.field =np.real(field_Sol)
        self.field_FFT = field_FFT_Sol


class PicSimulation:
    ##runs the simulation

    def __init__(self, particle_positions, particle_velocities, q, m, Xi, Xf, num_cells, dt,
                 save_field=False,save_FFT=False,save_pos=False,save_vel=False, order_den=0):
        ##initiates the simulation by making a particle array with the positions and velocities, same mass and charge, and defining the field.
        ##calculates the half value velocities and fields and saves the first data
        self.order_den=order_den
        self.dt=dt
        self.field = field(Xi, Xf, num_cells)
        if particle_positions.shape != particle_velocities.shape:
            print("velocities and positions not the same size")
        else:
            self.initialize_particles(particle_positions, particle_velocities, q, m)

        self.update_field()
        self.initialize_half_vel()

        self.save_field=save_field
        self.save_pos=save_pos
        self.save_FFT=save_FFT
        self.save_vel=save_vel

        self.data = np.empty((5,), dtype=object)
        self.data[0] = []
        self.data[1] = []
        self.data[2] = []
        self.data[3] =[]
        self.data[4] = []

        self.data[0].append(np.array(self.field.grid_pos))
        self.data[0].append(np.array(self.field.freqs))


        self.data[1].append(np.array(self.field.field))
        self.data[2].append(np.array(self.field.field_FFT))
        pos_array=np.empty((0,), dtype=float)
        for particle in self.particles:
            pos_array=np.append(pos_array,particle.x)
        self.data[3].append(np.array(pos_array))


        vel_array = np.empty((0,), dtype=float)
        for particle in self.particles:
            vel_array=np.append(vel_array,particle.v)
        self.data[4].append(np.array(vel_array))



    def initialize_particles(self, particle_positions, particle_velocities, q, m):
        ##puts in the particles and their posiotions, with equal mass and charges
        self.num_particle = len(particle_positions)
        self.particles = []
        for i in range(self.num_particle):
            self.particles.append(particle(particle_positions[i], particle_velocities[i], q, m))

    def update_field(self):
        ##updates the field by particle position
        if self.order_den == 0:
            density = self.field.compute_density_first_order_method(self.particles)
        else:
            if self.order_den == 1:
                density = self.field.compute_density_triangle(self.particle)
            else:
                density = self.field.compute_density_gaussian(self.particle)

        self.field.density_field(density)

    def initialize_half_vel(self):
        ##calculates and changes the velocoties at t=-dt/2
        for particle in self.particles:
            particle.update_vel(particle.applied_field(self.field),-1*self.dt/2)

    def update_vel(self):
        ##updates the velocites with weighting the fields to the particles
        for particle in self.particles:
            particle.update_vel(particle.applied_field(self.field),self.dt)

    def update_pos(self):
        ##updates the positions of the particles by their velocities
        for particle in self.particles:
            particle.update_pos(self.dt)

    def time_step(self):
        # propogates one time step and saves the data in data
        self.update_pos()
        self.update_field()
        self.update_vel()
        if self.save_field:
            self.data[1].append(np.array(self.field.field))
        if self.save_FFT:
            self.data[2].append(np.array(self.field.field_FFT))
        if self.save_pos:
            pos_array=np.empty((0,), dtype=float)
            for particle in self.particles:
                pos_array=np.append(pos_array,particle.x)

            self.data[3].append(np.array(pos_array))
        if self.save_vel:
            vel_array = np.empty((0,), dtype=float)
            for particle in self.particles:
                pos_array=np.append(pos_array,particle.v)

            self.data[4].append(np.array(vel_array))

    def run_simulation(self, num_steps):
        ##runs the simulation for num_steps times and return the data
        for i in range(num_steps):
            self.time_step()

        return self.data


num_particles = 3
particle_positions = np.linspace(0.1, 0.9, num_particles)  # Evenly spaced initial positions
particle_velocities = np.random.uniform(-0.05, 0.05, num_particles)  # Random velocities in range [-0.05, 0.05]

q, m = -0.00003, 0.02
Xi, Xf = -50.0, 50.0
num_cells = 500
dt = 0.00001

# Initialize the simulation
simulation = PicSimulation(
    particle_positions, particle_velocities, q, m, Xi, Xf, num_cells, dt,
    save_vel=True, save_pos=True
)

# Run simulation for 10 steps
results = simulation.run_simulation(100)

# Plot particle positions
plt.figure(figsize=(10, 6))
for t, positions in enumerate(results[3]):
    plt.scatter(positions, [t] * len(positions), label=f"Step {t}")
plt.xlabel("Position")
plt.ylabel("Time Step")
plt.title("Particle Positions Over Time (20 Particles)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
plt.tight_layout()
plt.show()
