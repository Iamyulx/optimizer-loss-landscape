import numpy as np
import matplotlib.pyplot as plt

from run_experiment import sgd_traj, adam_traj, adamw_traj


grid = np.linspace(-3, 3, 200)

X, Y = np.meshgrid(grid, grid)

Z = X**2 + Y**2 + 0.3*np.sin(3*X)*np.sin(3*Y)


plt.figure(figsize=(8,6))

plt.contour(X, Y, Z, levels=50)

plt.plot(sgd_traj[:,0], sgd_traj[:,1], label="SGD")
plt.plot(adam_traj[:,0], adam_traj[:,1], label="Adam")
plt.plot(adamw_traj[:,0], adamw_traj[:,1], label="AdamW")

plt.scatter([2.5],[2.5], label="Start")

plt.title("Optimizer Dynamics on Loss Landscape")
plt.xlabel("Parameter x")
plt.ylabel("Parameter y")

plt.legend()

plt.savefig("optimizer_trajectories.png", dpi=300, bbox_inches="tight")

plt.show()
