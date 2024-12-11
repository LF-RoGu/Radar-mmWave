from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Example data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z1 = np.sin(np.sqrt(X**2 + Y**2))
Z2 = np.cos(np.sqrt(X**2 + Y**2))

# Create figure and 3D subplots
fig = plt.figure(figsize=(14, 6))

# First 3D subplot
ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, 1st plot
ax1.plot_surface(X, Y, Z1, cmap='viridis')
ax1.set_title("3D Plot 1")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Second 3D subplot
ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, 2nd plot
ax2.plot_surface(X, Y, Z2, cmap='plasma')
ax2.set_title("3D Plot 2")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.tight_layout()
plt.show()
