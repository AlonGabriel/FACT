import abc

import numpy as np
from munch import Munch
import scipy.stats as stats
import torch
from ignite.engine import Engine
from ignite.utils import convert_tensor
from sklearn.cluster import DBSCAN

from transforms import IntensityAwareAugmentation


class BaseTrainer(abc.ABC):

    def __init__(self, model, criterion, optimizer, device, config):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.engine = Engine(self._update)
        self.config = Munch(**config)

    def _update(self, engine, batch):
        self.optimizer.zero_grad()
        self.model.train()
        batch = self.prepare_batch(batch)
        loss = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def prepare_batch(self, batch):
        return [convert_tensor(el, device=self.device) for el in batch]

    def on(self, event, handler, *args, **kwargs):
        return self.engine.add_event_handler(event, lambda engine: handler(*args, **kwargs))

    def run(self, data, num_epochs):
        return self.engine.run(data, num_epochs)

    @abc.abstractmethod
    def forward(self, batch):
        pass

    def state_dict(self):
        return {'model': self.model, 'criterion': self.criterion, 'optimizer': self.optimizer, 'engine': self.engine}


class Supervised(BaseTrainer):

    def forward(self, batch):
        x, y = batch
        outputs = self.model(x)
        outputs = getattr(outputs, self.config.target)
        return self.criterion(outputs, y)


