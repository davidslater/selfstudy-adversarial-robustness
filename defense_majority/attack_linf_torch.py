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

import numpy as np
import torch

import common.framework

from defense_majority.task_definition import LINF_THRESHOLD as eps

class LinfAttack(common.framework.Attack):

    def attack(self, model, x, y):
        def fail(x_adv, y_adv, stage=""):
            if stage:
                stage = stage + ": "
            x = x_adv.numpy()
            y = y_adv.numpy()
            detected = model.detect(x)
            correct = model.classify(x).argmax(axis=1) == y
            failures = correct | detected
            print(f"{stage}{failures.sum()} failures")
            return failures

        x_orig = torch.tensor(x)
        x = torch.tensor(x)
        y = torch.LongTensor(y)
        C = 10  # classes
        eps_step = eps / 2
        steps = 5
         
        failures = fail(x, y, "Benign")
        proxies = []
        for convnet in model.convnets:
            proxies.append(torch.nn.Sequential(*convnet.layers[:-1]))
        loss = torch.nn.CrossEntropyLoss()
        
        # try 9 targeted attacks
        x_full = x
        for j in range(1, C):
            print(f"Attacking class y + {j} mod {C}")
            x = x_orig[failures]
            y_target = (y[failures] + j) % C

            for i in range(steps):
                print(f"PGD step {i}")
                x.requires_grad = True
                for proxy in proxies:
                    loss(proxy(x), y_target).backward()
                x = x.detach() - torch.sign(x.grad) * eps_step
                x = torch.clip(x, 0.0, 1.0)
                x = torch.min(torch.max(x, x_orig[failures] - eps), x_orig[failures] + eps)
            
            x_full[failures] = x
            failures = fail(x_full, y)
            
        x = x_full
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
