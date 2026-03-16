import torch


class SGD:

    def __init__(self, lr=0.05):
        self.lr = lr

    def step(self, x, y, grad_fn):

        gx, gy = grad_fn(x, y)

        x = x - self.lr * gx
        y = y - self.lr * gy

        return x, y


class Adam:

    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mx = 0
        self.my = 0
        self.vx = 0
        self.vy = 0
        self.t = 0

    def step(self, x, y, grad_fn):

        gx, gy = grad_fn(x, y)

        self.t += 1

        self.mx = self.beta1*self.mx + (1-self.beta1)*gx
        self.my = self.beta1*self.my + (1-self.beta1)*gy

        self.vx = self.beta2*self.vx + (1-self.beta2)*(gx**2)
        self.vy = self.beta2*self.vy + (1-self.beta2)*(gy**2)

        mx_hat = self.mx/(1-self.beta1**self.t)
        my_hat = self.my/(1-self.beta1**self.t)

        vx_hat = self.vx/(1-self.beta2**self.t)
        vy_hat = self.vy/(1-self.beta2**self.t)

        x = x - self.lr * mx_hat / (torch.sqrt(vx_hat) + self.eps)
        y = y - self.lr * my_hat / (torch.sqrt(vy_hat) + self.eps)

        return x, y


class AdamW:

    def __init__(self, lr=0.05, weight_decay=0.01,
                 beta1=0.9, beta2=0.999, eps=1e-8):

        self.lr = lr
        self.wd = weight_decay

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mx = 0
        self.my = 0
        self.vx = 0
        self.vy = 0
        self.t = 0

    def step(self, x, y, grad_fn):

        gx, gy = grad_fn(x, y)

        self.t += 1

        # decoupled weight decay
        x = x * (1 - self.lr*self.wd)
        y = y * (1 - self.lr*self.wd)

        self.mx = self.beta1*self.mx + (1-self.beta1)*gx
        self.my = self.beta1*self.my + (1-self.beta1)*gy

        self.vx = self.beta2*self.vx + (1-self.beta2)*(gx**2)
        self.vy = self.beta2*self.vy + (1-self.beta2)*(gy**2)

        mx_hat = self.mx/(1-self.beta1**self.t)
        my_hat = self.my/(1-self.beta1**self.t)

        vx_hat = self.vx/(1-self.beta2**self.t)
        vy_hat = self.vy/(1-self.beta2**self.t)

        x = x - self.lr * mx_hat / (torch.sqrt(vx_hat)+self.eps)
        y = y - self.lr * my_hat / (torch.sqrt(vy_hat)+self.eps)

        return x, y
