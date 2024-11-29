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
        return iter(self.x, self.v, self.q, self.m)

    def update_pos(self, dt):
        ## moves the particle with it's current velocity for the time jump dt
        ## x_new = x + v*dt
        self.x += self.v * dt

    def update_vel(self, Field, dt):
        ## changes the velocity by a given field, for time jump dt
        ## v_new = v+ a * dt = v+E*q/m * dt
        acceleration = (Feild * self.q) / self.m
        self.v += acceleration * dt

    def applied_field(self,field):
        ##lineary calculates the field on the particle
        grid_pos, field_val, dx= field
        if self.x > grid.pos[-1] or self.x< grid.pos[0]:
            return 0.0
        nearest_grid_point = int(np.round(self.x / dx))
        field_applied=((self.x-grid_pos[nearest_grid_point])/dx)*field_val[nearest_grid_point]+((grid_pos[nearest_grid_point+1]-self.x)/dx)*field_val[nearest_grid_point+1]
        return field_applied



class field:
    ##class of the feild
    def __init__(self, Xi, Xf, num_cells):
        ## input of borders of the grid xi xf, and number of the cells.
        ## parameters: dx cell size
        ## field: for the feild (not used yet)
        ## grid_pos: list of grid point positions
        ## num celles: number of cells
        self.dx = (Xf - Xi) / num_cells
        self.grid_pos = np.linspace(Xi, Xf, num_cells)
        self.num_cells = num_cells

        self.field = np.zeros(num_cells)  # used?
        self.freqs = 2 * np.pi * np.fft.fftfreq(Xf - Xi, d=self.dx)
        self.field_FFT = np.zeros(num_cells)

    def __iter__(self):
        ## iter fun of feild only returns the positions and values
        return iter(self.grid_pos, self.field, dx)

    def compute_density_first_order_method(self, particles):
        ##Calculates charge density using 1st order method, taken to be the size of the grid cell
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
            Xiplusone = grid_pos[nearest_grid_point + 1]
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
            for cell in range(xleft, xright + 1):
                distance = abs(grid_pos[cell] - x)
                if distance <= width:
                    Ai, _ = quad(squar_func(x), self.grid_pos(cell), self.grid_pos(cell + 1))
                    density[cell] += Ai

        return density

    def compute_density_gaussian(self, particles):
        density = np.zeros(num_cells)
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
                Ai, _ = quad(gaussian_func(x), self.grid_pos(cell), self.grid_pos(cell + 1))
                density[cell] += Ai

        return density

    def density_feild(self, density):
        ## finds the electric potential and electrical field of the density on the grid
        ## puts the values in field and field_FFT
        epsilon = 8, 85419 * 10 ** (-12)  ##Units of ??

        density_FFT = np.fft.fft(density)
        FFT_freq = 2 * pi * np.array(np.linspace(-self.num_cells / 2, self.num_cells / 2 + 1, self.num_cells)) / (
                    self.dx * self.num_cells)

        a = density_freq * self.dx / 2
        potential_correction = density_freq * sin(a) / (a)
        potential_FFT = density_FFT / (epsilon * (potential_freq ** 2))

        b = 2 * a
        feild_correction = density_freq * sin(b) / (b)
        feild_FFT_Sol = -1 * np.j * feild_correction * potential_FFT
        feild_Sol = np.fft.ifft(feild_FFT)

        self.field = feild_Sol
        self.field_FFT = field_FFT_Sol


class PicSimulation:
    ##runs the simulation

    def __init__(self, particle_positions, particle_velocities, q, m, Xi, Xf, num_cells, dt,
                 save_field=false,save_FFT=false,save_pos=false,save_vel=false, order_den=0):
        ##initiates the simulation by making a particle array with the positions and velocities, same mass and charge, and defining the feild.
        ##calculates the half value velocities and fields and saves the first data
        self.order_den=order_den
        self.dt=dt
        self.field = field(Xi, Xf, num_cells)
        if particle_positions.shape != particle_velocities.shape:
            print("velocities and positions not the same size")
        else:
            self.particles = initialize_particles(particle_positions, particle_velocities, q, m)

        self.update_field()
        self.initialize_half_vel()

        self.save_field=save_field
        self.save_pos=save_pos
        self.save_FFT=save_FFT
        self.save_vel=save_vel

        self.data = np.array(size=5, dtype=np.array(dtype=float))
        self.data[0].append(self.field.grid_pos)
        self.data[0].append(self.field.freqs)


        data[1].append(self.field.field)
        data[2].append(self.field.field_FFT)
        pos_array=np.array(dtype=float)
        for particle in self.particles:
            pos_array.append(particle.x)
        data[3].append(pos_array)


        vel_array = np.array(dtype=float)
        for particle in self.particles:
            pos_array.append(particle.v)
        data[4].append(vel_array)



    def initialize_particles(self, particle_positions, particle_velocities, q, m):
        ##puts in the particles and their posiotions, with equal mass and charges
        self.num_particle = particle_positions.shape
        self.particles = np.array(dtype=particle)
        for i in range(num_particle):
            self.particles().append(particle(particle_positions(i), particle_velocities(i), q, m))

    def update_field(self):
        if self.order_den == 0:
            density = self.field.compute_density_first_order_method(self.particle)
        else:
            if self.order_den == 1:
                density = self.field.compute_density_triangle(self.particle)
            else:
                density = self.field.compute_density_gaussian(self.particle)

        self.field.density_feild(density)

    def initialize_half_vel(self):
        ##calculates and changes the velocoties at t=-dt/2
        for particle in self.particles:
            particle.update_vel(particle.applied_field(self.field),-self.dt/2)

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
            data[1].append(self.field.field)
        if self.save_FFT:
            data[2].append(self.field.field_FFT)
        if self.save_pos:
            pos_array=np.array(dtype=float)
            for particle in self.particles:
                pos_array.append(particle.x)

            data[3].append(pos_array)
        if self.save_vel:
            vel_array = np.array(dtype=float)
            for particle in self.particles:
                pos_array.append(particle.v)

            data[4].append(vel_array)

    def run_simulation(self, num_steps):
        for i in range(num_steps):
            self.time_step()

        return self.data
