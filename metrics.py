# noinspection PyUnresolvedReferences
from ignite.metrics import ROC_AUC
from ignite.metrics import Recall


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
