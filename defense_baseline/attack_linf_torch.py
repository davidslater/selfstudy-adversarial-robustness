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


def pgd(model, x, y, x_orig=None, eps=4/255, eps_step=2/255, steps=10):
    """
    x_orig - the actual original x, used for projecting back to linf ball
    """
    if x_orig is None:
        x_orig = torch.tensor(x)
    else:
        x_orig = torch.tensor(x_orig)

    x = torch.tensor(x)
    y = torch.LongTensor(y)
    loss = torch.nn.CrossEntropyLoss()
    for i in range(steps):
        print(i)
        x.requires_grad = True
        loss(model.convnet(x), y).backward()
        x = x.detach() + torch.sign(x.grad) * eps_step
        x = torch.max(torch.min(x, x_orig + eps), x_orig - eps)

        #model_prediction = model.convnet(xt)
        # TODO: check success after each iteration when calculating loss
        #y_pred = model_prediction.detach().numpy()
        #failures = (y_pred.argmax(axis=1) == y) | (np.isnan(y_pred).any(axis=1))

        #output = loss(model_prediction, yt)
        #output.backward()
        #grad_sign = torch.sign(xt.grad)
        #xt = xt.detach() + torch.sign(xt.grad) * eps_step
        
        # clip (torch cannot do tensor clipping)
        #xt = torch.max(torch.min(xt, x_orig + eps), x_orig - eps)
        #xt = xt.detach()  # disconnect from computational graph
        #xt.grad.zero_()
        #model.convnet.zero_grad()
    
    return x.numpy()
    #x = xt.detach().numpy()
    #return x


def fgsm(model, x, y, eps=4/255, eps_step=4/255):
    xt = torch.tensor(x, requires_grad=True)
    loss = torch.nn.CrossEntropyLoss()
    output = loss(model.convnet(xt), torch.LongTensor(y))
    output.backward()
    grad_sign = np.sign(xt.grad.numpy())
    x = x + grad_sign * eps_step
    return x


def uniform_linf_noise(x, eps=4/255):
    x_out = x + 2*(np.random.random(x.shape) - 0.5)*eps
    return x_out.astype(x.dtype)


def max_linf_noise(x, eps=4/255):
    x_out = np.sign(np.random.random(x.shape) - 0.5)*eps + x
    return x_out.astype(x.dtype)


def failed(model, x_adv, y):
    y_pred, _ = model(x_adv)
    failures = (y_pred.argmax(axis=1) == y) | (np.isnan(y_pred).any(axis=1))
    return failures


class LinfAttack(common.framework.Attack):

    def attack(self, model, x, y):
        eps = 4/255
        eps_step = eps/2
        pgd_kwargs = dict(
            eps=eps,
            eps_step=eps_step,
            steps=20,
        )

        # PGD-10 restarts
        random_restarts = 5
        print(f"Initial size: {len(x)}")
        x_adv = x.copy()
        index = failed(model, x_adv, y)
        print(f"After benign: {index.sum()}")
        if not index.any():
            return x_adv

        x_adv[index] = pgd(model, x[index], y[index], **pgd_kwargs)
        index[index] = failed(model, x_adv[index], y[index])
        print(f"After initial PGD: {index.sum()}")
        for i in range(random_restarts):
            if not index.any():
                return x_adv
           
            x_noise = uniform_linf_noise(x[index], eps=4/255)
            x_adv[index] = pgd(model, x_noise, y[index], x_orig=x[index], **pgd_kwargs)
            index[index] = failed(model, x_adv[index], y[index])
            print(f"After random restart {i+1}: {index.sum()}")
        
        # PGD-10
        # x_adv = pgd(model, x, y, eps=4/255, eps_step=2/255, steps=10) 

        # FGSM
        # x_adv = fgsm(model, x, y, eps=4/255, eps_step=4/255)

        # Add noise within epsilon ball
        # x_adv = max_linf_noise(x, eps=4/255)
        

        return x_adv

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
