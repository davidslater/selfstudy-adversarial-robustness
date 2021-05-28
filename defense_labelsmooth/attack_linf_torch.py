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

from defense_labelsmooth.task_definition import LINF_THRESHOLD as eps

import torch 

def pgd_step(model, loss, x, y, eps, eps_step, x_orig, targeted=False):
    x.requires_grad = True
    loss(model(x), y).backward()
    if targeted:
        x = x.detach() - torch.sign(x.grad) * eps_step
    else:
        x = x.detach() + torch.sign(x.grad) * eps_step
    x = torch.max(torch.min(x, x_orig + eps), x_orig - eps)
    return x


class LinfAttack(common.framework.Attack):

    def attack(self, model, x, y):
        x_orig = torch.tensor(x)
        x = torch.tensor(x)
        y = torch.LongTensor(y)
        proxy = torch.nn.Sequential(*model.convnet.layers[:-1])
        loss = torch.nn.CrossEntropyLoss()
        steps = 5
        eps_step = eps / 2

        # Untargeted PGD
        failures = model.convnet(x).argmax(axis=1) == y
        print(f"Benign: failures = {failures.sum()}")
        for i in range(steps):
            if not any(failures):
                break

            x[failures] = pgd_step(proxy, loss, x[failures], y[failures], eps, eps_step, x_orig[failures], targeted=False)
            failures = model.convnet(x).argmax(axis=1) == y
            print(f"Untargeted PGD step {i}: failures = {failures.sum()}")

        # Targeted PGD
        for j in range(1, 10):
            y_target = (y + j) % 10
            for i in range(steps):
                if not any(failures):
                    break

                print(f"PGD, step {i}")
                x[failures] = pgd_step(proxy, loss, x[failures], y_target[failures], eps, eps_step, x_orig[failures], targeted=True)
                failures = model.convnet(x).argmax(axis=1) == y
                print(f"Target {j} PGD step {i}: failures = {failures.sum()}")


#         for j in range(1, 10):
#             if not any(failures):
#                 break
#             # NOTE: FGSM with multiple targets goes down to 5/100 with eps = 8/255
#             y_target = (y + 2) % 10
#             x.requires_grad = True
#             output = proxy(x)
#             #print(output)
#             #l = loss(output, y)
#             l = loss(output, y_target)
#             print(l)
#             l.backward()
#             #x = x.detach() + torch.sign(x.grad) * eps
#             x = x.detach() - torch.sign(x.grad) * eps
#             break
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
