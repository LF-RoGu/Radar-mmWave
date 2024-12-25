import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def estimate_velocity(P_prev, P_curr, delta_t):
    """
    Estimate the linear velocity vector 'v' that best explains
    the displacement from P_prev to P_curr in a least-squares sense.
    
    P_prev: (N,3) array of points at time (t-1)
    P_curr: (N,3) array of points at time t
    delta_t: float, time difference between frames
    
    Returns: v_est (3,) numpy array
    """
    # We assume each row i in P_prev corresponds to row i in P_curr.
    # Our model: P_curr[i] ~ P_prev[i] + v * delta_t
    
    # Build the system: P_curr[i] - P_prev[i] = v * delta_t
    # => (P_curr[i] - P_prev[i]) / delta_t = v
    # But we want a single v that works best for all i in a least-squares sense.

    # A simple direct approach:
    # For each i: P_curr[i] - P_prev[i] = delta_t * v
    # => P_curr[i] - P_prev[i] = delta_t * v
    # => rearrange => (1/delta_t) * (P_curr[i] - P_prev[i]) = v
    # If there's noise, we do it in a least-squares sense:
    # "A * v = b" form, where A is Nx3, v is 3x1, b is Nx1 (but we have 3 dims).
    
    N = P_prev.shape[0]
    # Construct A as block for each dimension. But we can do it more simply:
    # We want v to satisfy P_curr[i] - P_prev[i] = delta_t * v.
    # Let's define D[i] = P_curr[i] - P_prev[i].
    
    D = P_curr - P_prev  # shape (N,3)
    # So we want D[i] ~ delta_t * v, or D = delta_t * 1 * v => D / delta_t = v
    # In vector form, we want v that solves:
    # D = delta_t * (1_N v^T) ??? 
    # Actually, an easier approach: we can just average D/delta_t because
    # if there's no rotation, the best velocity is the average displacement / delta_t.
    
    v_est = np.mean(D, axis=0) / delta_t
    
    return v_est

def main():
    # Example usage
    
    # 1) Generate synthetic data for demonstration:
    # Let's create 100 random points for the 'previous frame'
    N = 100
    P_prev = np.random.uniform(-10, 10, size=(N, 3))
    
    # True velocity
    true_v = np.array([1.5, -0.5, 0.8])  # (m/s) for example
    delta_t = 0.1  # 100 ms between frames
    
    # 2) Create the current frame by applying the true velocity
    # ignoring rotation in this simple demonstration
    P_curr = P_prev + true_v * delta_t
    
    # Optionally, add some noise
    noise_std = 0.05
    P_curr += np.random.normal(0, noise_std, size=P_curr.shape)
    
    # 3) Estimate velocity from the two frames
    v_est = estimate_velocity(P_prev, P_curr, delta_t)
    
    print("True velocity: ", true_v)
    print("Estimated velocity: ", v_est)
    
    # 4) (Optional) Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points from previous frame
    ax.scatter(P_prev[:, 0], P_prev[:, 1], P_prev[:, 2], c='blue', label='Frame t-1')
    # Plot the points from current frame
    ax.scatter(P_curr[:, 0], P_curr[:, 1], P_curr[:, 2], c='red', label='Frame t')
    
    ax.set_title('Pose Estimation (Simple Velocity) Demo')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
