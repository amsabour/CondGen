import torch
from torch import nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        """
        outputs = outputs[0]
        loss = self.loss(outputs, targets)
        accuracy = self._calculate_accuracy(outputs, targets)
        return loss, accuracy

    def _get_correct(self, outputs):
        raise NotImplementedError()

    def _calculate_accuracy(self, outputs, targets):
        correct = self._get_correct(outputs)
        return 100. * (correct == targets).sum().float() / targets.size(0)


class BinaryClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None):
        super().__init__()
        if reduction is not None:
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def _get_correct(self, outputs):
        return outputs > 0.5


class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, weight=None, reduction=None):
        super().__init__()
        if reduction is not None:
            self.loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        else:
            self.loss = nn.CrossEntropyLoss(weight=weight)

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)
