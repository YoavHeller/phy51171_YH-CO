import numpy as np

particle (x,v,q,m)
from scipy.integrate import quad

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
