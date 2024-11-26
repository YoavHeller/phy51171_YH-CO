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

    def update_pos(self, dt):
        ## moves the particle with it's current velocity for the time jump dt
        self.x += self.v * dt

    def update_vel(self, Field, dt):
        ## changes the velocity by a given force, for time jump dt
        ##calculated by newton
        acceleration = (Feild * self.q) / self.m
        self.v += acceleration * dt


class field:
    ##class of the feild
    def __init__(self, Xi, Xf, num_cells):
        ## input of borders of the grid xi xf, and number of the cells.
        self.dx = (Xf - Xi) / num_cells
        self.field = np.zeros(num_cells) #used?
        self.grid_pos = np.linspace(Xi, Xf, num_cells)
        self.num_cells=num_cells

    def compute_density_first_order_method(self, particles):
        ##Calculates particles density using 1st order method, taken to be the size of the grid cell
        ## returns the density as an array in the grids positions
        density = np.zeros(self.num_cells)
        for par in particles:
            (x, v, q, m) = par
            # Find the nearest grid point
            nearest_grid_point = int(np.round(x / self.dx))
            if nearest_grid_point < 0 or nearest_grid_point >= self.num_cells:
                continue
            Xi = grid_pos[nearest_grid_point]
            density[nearest_grid_point] += (q / self.dx) * (x - Xi + self.dx)
            Xiplusone = grid_pos[nearest_grid_point+1]
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

                    return lambda y: (abs(y-x)<=(width/2))*q/width

            # Define the range of cells to influence (±2 cells around nearest point)
            cellleft = max(int(np.floor((x - width / 2) / self.dx)), 0)
            cellright = min(int(np.floor((x + width / 2) / self.dx)), self.num_cells - 1)  # why use floor?
            for cell in range(xleft, xright + 1):
                distance = abs(grid_pos[cell] - x)
                if distance <= width:
                    Ai, _ = quad(squar_func(x), self.grid_pos(cell), self.grid_pos(cell + 1))
                    density[cell] += Ai

        return density

    def compute_density_gaussian_density(self, particles):
        density = np.zeros(num_cells)
        sigma=2*self.dx
        def gaussian_func(x0):
            # Normalization factor for Gaussian distribution
            A = q / (np.sqrt(2 * np.pi) * sigma)
            return lambda x:  A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

        for par in particles:
            (x, v, q, m) = par
            nearest_grid_point = int(np.round(x / dx))
            # Define bounds for integration based on Gaussian spread
            num_points_to_cover = int(3 * sigma / self.dx)  # Cover 3 sigma range for significant spread
            xleft = max(nearest_grid_point - num_points_to_cover, 0)
            xright = min(nearest_grid_point + num_points_to_cover, self.num_cells - 1)
            for cell in range(xleft, xright + 1):
                Ai, _ = quad(gaussian_func(x), self.grid_pos(cell),self.grid_pos(cell+1))
                density[cell] += Ai


        return density


class PicSimulation:
    ##runs the simulation
    def PicSimulation(self, particle_positions, particle_velocities, q, m, Xi, Xf, num_cells):
        ##initiates the simulation by making a particle array with the positions and velocities, same mass and charge, and defining the feild.
        self.field = field(Xi, Xf, num_cells)
        if particle_positions.shape != particle_velocities.shape:
            print("velocities and positions not the same size")
        else:
            self_particles = initialize_particles(particle_positions, particle_velocities, q, m)

    def initialize_particles(self, particle_positions, particle_velocities, q, m):
        self.num_particle = particle_positions.shape
        self.particles = np.array(dtype=particle)
        for i in range(num_particle):
            self.particles().append(particle(particle_positions(i), particle_velocities(i), q, m))

    def solve_poisson(self, particles, k):
        epsilon = 8, 85419 * 10 ** (-12)
        density = compute_density_first_order_method(self, particles)
        density_k = np.fft.fft(density)
        phi_k = density_k / (epsilon * k ** 2)
        phi = np.fft.ifft(phi_k).real
        E = np.zeros(num_cells)
        E[0] = -phi[1] / (2 * self.dx)
        for i in range(1, num_cells - 1):
            E[i] = phi[i - 1] - phi[i + 1] / (2 * self.dx)
        E[num_cells] = phi[num_cells - 1] / (2 * self.dx)
        return E

       def time_step(self,particles,dt,Xi, Xf):
        density = field.compute_density_first_order_method(self,particles)
        k = 2*np.pi*np.fft.fftfreq(Xf-Xi, d=self.dx)
        electric_fields = PicSimulation.solve_poisson(density, k)

        for particle in particles:
            (x, v, q, m) = particle
            cell = int(np.round(x / self.dx))
            E = electric_fields[cell]
            particle.update_vel(self,E, dt)

        for particle in particles:
            particle.update_pos(self,dt)

