import torch as th
from torch.optim.optimizer import Optimizer, required
from utils import *
import os
import math

class RiemannianSGD(Optimizer):
    """Riemannian stochastic gradient descent.
    """
    def __init__(self, args, params, lr):
        defaults = dict(lr=lr)
        self.args = args
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """
        Performs a single optimization step.
        """
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p = self.args.weight_manifold.rgrad(p, d_p)
                if lr is None:
                    lr = group['lr']
                p.data = self.args.weight_manifold.exp_map_x(p, -lr * d_p)
        return loss
