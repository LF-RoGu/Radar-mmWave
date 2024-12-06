import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate example data: Cosine from -90 to 90 degrees
phi = np.linspace(-90, 90, 50)  # angles in degrees
phi_rad = np.radians(phi)  # convert to radians
radial_speed = np.cos(phi_rad) + np.random.normal(0, 0.1, phi_rad.shape)  # Cosine with noise

# Define a model function (cubic: f(phi) = a*phi^3 + b*phi^2 + c*phi + d)
def model(params, x):
    a, b, c, d = params
    return a * x**3 + b * x**2 + c * x + d

# Define the objective function (sum of absolute residuals)
def objective(params, x, y):
    predictions = model(params, x)
    residuals = np.abs(y - predictions)
    return np.sum(residuals)

# Initial guess for parameters (e.g., cubic coefficients)
initial_guess = [0.0, 0.0, 0.0, 0.0]

# Optimize the parameters to minimize the absolute residuals
result = minimize(objective, initial_guess, args=(phi, radial_speed), method='Powell')

# Extract the optimized parameters
optimized_params = result.x
print("Optimized Parameters:", optimized_params)

# Generate the fitted curve
phi_fit = np.linspace(np.min(phi), np.max(phi), 100)
radial_speed_fit = model(optimized_params, phi_fit)

# Plot the data and the fitted curve
plt.figure(figsize=(8, 6))
plt.scatter(phi, radial_speed, color='blue', label='Data Points (Cosine with Noise)', zorder=5)
plt.plot(phi_fit, radial_speed_fit, color='red', label='Fitted Curve (Cubic)', linewidth=2)
plt.xlabel(r'$\phi$ (degrees)')
plt.ylabel('Radial Speed')
plt.title('Absolute Least Squares Fit (Cubic Model for Cosine Data)')
plt.legend()
plt.grid(True)
plt.xlim(-100, 100)
plt.ylim(-1.5, 1.5)
plt.show()
