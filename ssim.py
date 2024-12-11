from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.regression import *
from ignite.utils import *

# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests

param_tensor = torch.zeros([1], dtype=torch.complex64, requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

# create default model for doctests

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

manual_seed(666)

metric = SSIM(data_range=1.0)
metric.attach(default_evaluator, 'ssim')
preds = torch.rand([4, 3, 16, 16]).to(torch.complex64)
target = preds * 0.75
preds = (preds - preds.mean()) / preds.std()
target = (target - target.mean()) / target.std()
print(preds[0][0][0])
print(target[0][0][0])
state = default_evaluator.run([[preds, target]])
print(state.metrics['ssim'])