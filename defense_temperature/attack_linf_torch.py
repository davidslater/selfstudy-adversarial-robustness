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

from defense_temperature.task_definition import LINF_THRESHOLD as EPS


class LinfAttack(common.framework.Attack):

    def attack(self, model, x, y, eps=EPS, steps=10, restarts=5):
        proxy = torch.nn.Sequential(*model.convnet.layers[:-1])
        
        x_orig = torch.tensor(x)
        x = torch.tensor(x)
        y = torch.LongTensor(y)
        alpha = 0.000001
        loss = torch.nn.CrossEntropyLoss()
        eps_step = eps / 2
       
        # Test Benign
        output = proxy(x)
        failures = output.argmax(axis=1) == y

        x_full = x
        y_full = y
        for i in range(steps * (restarts + 1)):
            print("Failures: ", failures.sum().numpy())
            if not any(failures):
                break
            
            x = x_full[failures]
            y = y_full[failures]
            if i % steps == 0 and i > 0:
                # add random restart
                print("Restart")
                x = x_orig[failures] + 2*eps*(torch.rand(x.shape) - 0.5)

            x.requires_grad = True
            output = alpha * proxy(x)
            loss(output, y).backward()
            print(f"step {i}: output L1 = {output.abs().mean().detach().numpy()}, grad L1 = {x.grad.abs().mean().detach().numpy()}")
            x = x.detach() + torch.sign(x.grad) * eps_step
            x = torch.max(torch.min(x, x_orig[failures] + eps), x_orig[failures] - eps)
            x_full[failures] = x

            # check for failures
            failures[failures.clone()] = proxy(x).argmax(axis=1) == y

        x = x_full
        # for i in range(steps)
        #     x.requires_grad = True
        #     output = alpha * proxy(x)
        #     loss(output, y).backward()
        #     print(f"step {i}: output L1 = {output.abs().mean().detach().numpy()}, grad L1 = {x.grad.abs().mean().detach().numpy()}")
        #     x = x.detach() + torch.sign(x.grad) * eps_step
        #     x = torch.max(torch.min(x, x_orig + eps), x_orig - eps)
        
        

        #loss(alpha * output, y).backward()
        #for i in range(len(x)):
        #    output = model.convnet(x[i:i+1])
        #    l = loss(model.convnet(x[i:i+1]), y[i:i+1])
        #    l2 = loss2(model.convnet(x[i:i+1]), y[i:i+1])
        #    print(i, l.detach(), l2.detach(), output.detach(), y[i])
        #loss(model.convnet(x), y).backward()
        
        #print(x.grad.mean(), x.grad.std())

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
