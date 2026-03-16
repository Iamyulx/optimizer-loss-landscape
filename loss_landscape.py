import torch


def loss_fn(x, y):

    return x**2 + y**2 + 0.3 * torch.sin(3*x) * torch.sin(3*y)


def grad(x, y):

    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)

    loss = loss_fn(x, y)
    loss.backward()

    return x.grad, y.grad
