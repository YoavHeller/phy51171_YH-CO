import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from numba import jit




class particle:
## class of a particle


   def __init__(self, x, v, q, m):
   ## recives values for location (x), velocity (v), charge(q), and mass (m)
       self.x=x
       self.v=v
       self.q=q
       self.m=m


   def update_pos(self, dt):
   ## moves the particle with it's current velocity for the time jump dt
       self.x += self.v*dt


   def update_vel(self, Field, dt):
   ## changes the velocity by a given force, for time jump dt
   ##calculated by newton
       acceleration=(Feild*self.q)/self.m
       self.v += acceleration*dt


class field:
##class of the feild
   def __init__(self, Xi, Xf, num_cells):
       self.dx= (Xf-Xi)/num_cells
       self.field=np.zeros(num_cells)
       self.grid_pos= np.linspace(Xi, Xf, num_cells)
	def compute_density_first_order_method(self,particles):
       	self.particles = particles
       	density = np.zeros(num_cells)
       	for par in particles:
           		par = (x,v,q,m)
       # Find the nearest grid point
           nearest_grid_point = int(np.round(x/dx))
           Xi = grid_pos[nearest_grid_point]
           density[nearest_grid_point] += (q/dx)*(x - Xi + dx)
           Xiplusone = grid_pos[nearest_grid_point]
           density[nearest_grid_point + 1] += (q/dx)*(Xiplusone - x)
       return density
      
   def compute_density_triangle(self,particles):
       density = np.zeros(num_cells)
       for par in particles:
           (x, v, q, m) = par
           nearest_grid_point = int(np.round(x / self.dx))
           width = 2*dx
           # Define the range of cells to influence (±2 cells around nearest point)
           xleft = max(int(np.floor(x - width / 2*dx)), 0)
           xright = min(int(np.floor(x + width /2*dx)), self.num_cells - 1) #why use floor?


           for cell in range(xleft, xright + 1):
               distance = abs(grid_pos[cell] - x)
               if distance < width:
                   density[cell] += (q/dx)*(1 - distance)/width
      
       return density






   def compute_density_gaussian_density(self,particles):
       self.particles = particles
       density = np.zeros(num_cells)
        for par in particles:
          (x,v,q,m) = par
           # Normalization factor for Gaussian distribution
           A = q / (np.sqrt(2 * np.pi) * sigma)


           def gaussian_func(x):
               return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
          
           # Define bounds for integration based on Gaussian spread
           num_points_to_cover = int(3 * sigma / self.dx)  # Cover 3 sigma range for significant spread
           xleft = max(nearest_grid_point - num_points_to_cover, 0)
           xright = min(nearest_grid_point + num_points_to_cover, self.num_cells - 1)
           Ai, _ = quad (gaussian_func, xleft, xright)
           nearest_grid_point = int(np.round(x/dx))
           density[nearest_grid_point] += Ai/A
       return density


class PicSimulation:
##runs the simulation
   def PicSimulation(self, particle_positions, particle_velocities, q,m, Xi, Xf, num_cells):
   ##initiates the simulation by making a particle array with the positions and velocities, same mass and charge, and defining the feild.
       self.field=field(Xi, Xf, num_cells)
       if particle_positions.shape != particle_velocities.shape:
           print("velocities and positions not the same size")
       else:
           self_particles=initialize_particles(particle_positions, particle_velocities, q,m)




   def initialize_particles(self,particle_positions, particle_velocities, q,m):
           self.num_particle= particle_positions.shape
           self.particles= np.array(dtype=particle)
           for i in range(num_particle):
               self.particles().append(particle(particle_positions(i),particle_velocities(i),q,m))
   def solve_poisson:
       #bluh bluh
   def time_step:
       #bluh bluh



