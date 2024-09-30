# noinspection PyUnresolvedReferences
from ignite.metrics import (
    Metric,
    ROC_AUC,
    Recall,
)
from sklearn import metrics


class Sensitivity(Recall):
    pass  # Nothing to do! Recall and sensitivity are the same thing.


class Specificity(Sensitivity):

    def _prepare_output(self, output):
        # If we flip the labels, the recall formula will calculate the specificity.
        y_pred, y = output
        y_pred, y = 1 - y_pred, 1 - y
        return super()._prepare_output((y_pred, y))


def BalancedAccuracy(*args, **kwargs):
    return (Sensitivity(*args, **kwargs) + Specificity(*args, **kwargs)) / 2


class Silhouette(Metric):

    def __init__(self, output_transform=lambda x: x):
        super().__init__(output_transform=output_transform)
        self._Z, self._y = [], []

    def reset(self):
        self._Z, self._y = [], []

    def update(self, output):
        Z, y = output
        Z = Z.cpu().numpy()
        y = y.cpu().numpy()
        self._Z.extend(Z)
        self._y.extend(y)

    def compute(self):
        return metrics.silhouette_score(self._Z, self._y)

