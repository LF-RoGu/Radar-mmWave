import numpy as np
import matplotlib.pyplot as plt

def create_grid_map(coordinates, grid_size=(100, 100)):
    # Extract x and y coordinates
    x_coords, y_coords = zip(*coordinates)
    
    # Create 2D histogram
    histogram, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=grid_size)
    
    # Plotting the grid map
    plt.imshow(histogram.T, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Number of points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Grid Map based on Point Density')
    plt.show()

# Example usage
coordinates = [(2.5, 3.1), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (2.5, 3.0)]
create_grid_map(coordinates, grid_size=(10, 10))
