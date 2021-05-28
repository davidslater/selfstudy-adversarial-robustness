# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Placeholder for L_{inf} attack."""

import common.framework

import numpy as np
import torch
from defense_mergebinary.task_definition import LINF_THRESHOLD as EPS

class Proxy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.class_nets = model.class_nets

    def forward(self, x):
        predictions = [net(x) for net in self.class_nets]
        diff = torch.cat([p[:, 1:2] - p[:, 0:1] for p in predictions], dim=1)
        return diff


class SigmoidLoss:
    def __call__(self, y_pred, y):
        """
        Correct class should be >= 1
        Incorrect class should be <= -1
        """

        loss = 0
        for i, (yi_pred, yi) in enumerate(zip(y_pred, y)):
            true_label = yi_pred[yi]
            false_label = torch.cat([yi_pred[:yi], yi_pred[yi+1:]])
            loss = loss + (1 - torch.min(true_label, torch.FloatTensor([1])))
            loss = loss + (1 + torch.max(false_label.max(), torch.FloatTensor([-1])))
            # loss = loss + (len(false_label) + torch.max(false_label, torch.FloatTensor([-1])))
        return loss.sum()


class LinfAttack(common.framework.Attack):

    def attack(self, model, x, y, eps=EPS, steps=2):
        eps_step = eps/2
        x_orig = torch.tensor(x)
        x = torch.tensor(x)
        y = torch.LongTensor(y)
        loss = torch.nn.CrossEntropyLoss()
        proxy = Proxy(model)

        sloss = SigmoidLoss()
        weight = 1
        x_full = x
        y_full = y
        for j in range(1, 10):
            output = proxy(x_full)
            failures = (output.argmax(axis=1) == y_full) | (output.max(axis=1)[0] < 0.7)
            if not any(failures):
                break
            print(f"failures: {failures.sum().numpy()}")
            x = x_full[failures]
            x_full[failures] = x
            y = y_full[failures]
            y_target = (y + j) % 10
            #weight = 0.1
            for i in range(steps):
                print(f"step {i}")
                x.requires_grad = True
                output = proxy(x)

                y_target = (y + 8) % 10
                #(loss(output, y_target)).backward()
                #print(loss(output, y_target))
                print(sloss(output, y_target))
                #print(weight * sloss(output, y) + loss(output, y))

                (weight * sloss(output, y_target) + loss(output, y_target)).backward()
                x = x.detach() - torch.sign(x.grad) * eps_step
                x = torch.max(torch.min(x, x_orig[failures] + eps), x_orig[failures] - eps)

            x_full[failures] = x
        x = x_full
        
        predictions = model.classify(x.numpy())
        proxy_predictions = proxy(x)
        for p, prox in zip(predictions, proxy_predictions):
            if np.max(p) < 0.7:
                print(np.max(p) < 0.7, p, prox)
        return x.numpy()

# # If writing attack which operates on batch of examples is too complicated
# # then remove LinfAttack and uncommend LinfAttackNonBatched from below:
#
# class LinfAttackNonBatched(common.framework.NonBatchedAttack):
#
#     def attack_one_example(self, model, x,  y):
#         # TODO: Write your attack code here
#         # You can query model by calling `model(x)`
#
#         return x
