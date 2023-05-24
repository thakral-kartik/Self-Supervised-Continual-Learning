# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

import torch

import config
import ssl_models as models
import ssl_dataloader as dataloader


#main
if config.dataset == 'cifar100':
	trainloader, testloader = dataloader.load_cifar100()
elif config.dataset == 'cifar10':
	trainloader, testloader = dataloader.load_cifar10()
else:
	print("unknown dataset")
print("Dataset loaded")

model = models.Network()
model = model.to(config.device)

if config.load_model == True:
    model = model.load_model()


model.fit(trainloader, testloader)
model.save_model()


print("Done")
