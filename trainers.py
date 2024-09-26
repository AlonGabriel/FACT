import abc

import torch
from ignite.engine import Engine
from ignite.utils import convert_tensor
from munch import Munch

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


class FixMatch(BaseTrainer):

    def __init__(self, model, criterion, optimizer, device, config):
        super().__init__(model, criterion, optimizer, device, config)
        self.transform = IntensityAwareAugmentation()

    def forward(self, batch):
        (x, y), u = batch
        # Loss for labeled data
        outputs = self.model(x)
        outputs = outputs.predictions
        l_s = self.criterion(outputs, y)
        # Pseudo-labeling
        with torch.no_grad():
            outputs = self.model(u)
            outputs = outputs.predictions
            scores = torch.softmax(outputs, dim=1)
            scores, pseudo_labels = scores.max(dim=1)
            is_confident = scores.ge(self.config.tau).bool()
        # Update metrics
        self.smoothed_metric('num_confident', is_confident.sum().item())
        # Loss for unlabeled data
        l_u = 0
        if is_confident.any():
            ut = self.transform(u)
            outputs = self.model(ut)
            outputs = outputs.predictions
            l_u = self.criterion(outputs[is_confident], pseudo_labels[is_confident])
        # Weighted sum
        return l_s + self.config.llambda * l_u

    def smoothed_metric(self, metric, value, alpha=0.8):
        prev = self.engine.state.metrics.get(metric, value)
        updated = alpha * prev + (1 - alpha) * value
        self.engine.state.metrics[metric] = updated
