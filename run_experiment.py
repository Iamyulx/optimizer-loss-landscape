import torch
import numpy as np

from optimizers import SGD, Adam, AdamW
from loss_landscape import grad


def run_optimizer(opt, steps=100):

    x = torch.tensor(2.5)
    y = torch.tensor(2.5)

    trajectory = []

    for _ in range(steps):

        trajectory.append((x.item(), y.item()))

        x, y = opt.step(x, y, grad)

    return np.array(trajectory)


sgd_traj = run_optimizer(SGD())
adam_traj = run_optimizer(Adam())
adamw_traj = run_optimizer(AdamW())
