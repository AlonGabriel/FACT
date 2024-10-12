import abc

import torch
from ignite.engine import Engine
from ignite.metrics import Loss

from metrics import (
    BalancedAccuracy,
    Sensitivity,
    Specificity,
    ROC_AUC,
    Silhouette,
)
from utils import prepare_batch


class BaseEvaluatorFactory(abc.ABC):

    def __init__(self, model, criterion, augmenter, device):
        self.model = model
        self.criterion = criterion
        self.augmenter = augmenter
        self.device = device

    def _process(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, self.device)
            return self.process(batch)

    @abc.abstractmethod
    def process(self, batch):
        pass

    def register_metrics(self, engine):
        pass

    def create_engine(self):
        engine = Engine(self._process)
        Loss(self.criterion).attach(engine, 'loss')
        self.register_metrics(engine)
        return engine


class Classification(BaseEvaluatorFactory):

    def process(self, batch):
        x, y = batch
        outputs = self.model(x)
        outputs = outputs.predictions
        return outputs, y

    def register_metrics(self, engine):
        def apply_softmax(raw_scores=False):
            def wrapped(output):
                logits, y = output
                scores = logits.softmax(dim=1)
                if raw_scores:
                    return scores[:, 1], y  # Positives
                labels = torch.argmax(logits, dim=1)
                return labels, y

            return wrapped

        BalancedAccuracy(output_transform=apply_softmax()).attach(engine, 'balanced_accuracy')
        Sensitivity(output_transform=apply_softmax()).attach(engine, 'sensitivity')
        Specificity(output_transform=apply_softmax()).attach(engine, 'specificity')
        ROC_AUC(output_transform=apply_softmax(raw_scores=True)).attach(engine, 'auroc')


class Clustering(BaseEvaluatorFactory):

    def process(self, batch):
        x, y = batch
        outputs = self.model(x)
        outputs = outputs.embeddings
        return outputs, y

    def register_metrics(self, engine):
        Silhouette().attach(engine, 'silhouette')


class SimCLR(BaseEvaluatorFactory):

    def process(self, batch):
        x, y = batch
        x1, x2 = self.augmenter(x), self.augmenter(x)
        z1, z2 = self.model(x1), self.model(x2)
        z1, z2 = z1.embeddings, z2.embeddings
        return z1, z2
