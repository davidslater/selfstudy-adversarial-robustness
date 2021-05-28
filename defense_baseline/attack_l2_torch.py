
import common.framework

from defense_baseline.task_definition import L2_THRESHOLD as eps

import numpy as np
import torch

class L2Attack(common.framework.Attack):

    def attack(self, model, x, y):
        print(f"eps = {eps}")
        x = torch.tensor(x)
        noise = torch.randn(x.shape)
        l2_distance = torch.sqrt((noise**2).sum(axis=(1,2,3), keepdim=True))
        noise = (noise / l2_distance)*eps

        x = x + noise
        return x.numpy()

