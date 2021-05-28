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

import torch

from defense_blur import task_definition
EPS = task_definition.LINF_THRESHOLD

def blur(x):
    """
    Implement blur defense in torch
    """
    x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1))
    x_pad = (x_pad[:, :, :1] + x_pad[:, :, :-1])/2
    x_pad = (x_pad[:, :, :, :1] + x_pad[:, :, :, :-1])/2
    return x_pad


def project(x, x_orig, eps=EPS):
    return torch.max(torch.min(x, x_orig + eps), x_orig - eps)


class LinfAttack(common.framework.Attack):

    def attack(self, model, x, y, eps=EPS, steps=20):
        x_orig = torch.tensor(x)
        x = torch.tensor(x)
        y = torch.LongTensor(y)
        loss = torch.nn.CrossEntropyLoss()

        for i in range(steps):
            print(i)
            x.requires_grad = True
            loss(model.convnet(blur(x)), y).backward()
            x = x.detach() + torch.sign(x.grad) * eps
            x = project(x, x_orig, eps=EPS)
            x = torch.clip(x, 0.0, 1.0)

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
