import logging

import torch
import torch.nn.functional as F
from torch import nn


logging.basicConfig()
logger = logging.getLogger('losses')
logger.setLevel(logging.DEBUG)


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, batch_avg=True, batch_weight=None,
                 class_avg=True, class_weight=None, **kwargs):
        """

        Parameters
        ----------
        batch_avg:
            Whether to average over the batch dimension.
        batch_weight:
            Batch samples importance coefficients.
        class_avg:
            Whether to average over the class dimension.
        class_weight:
            Classes importance coefficients.
        """
        super().__init__()
        self.num_classes = num_classes
        self.batch_avg = batch_avg
        self.class_avg = class_avg
        self.batch_weight = batch_weight
        self.class_weight = class_weight
        logger.warning('Redundant loss function arguments:\n{}'
                       .format(repr(kwargs)))
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, input_, target, **kwargs):
        """

        Parameters
        ----------
        input_: (b, ch, d0, d1) tensor
        target: (b, d0, d1) tensor

        Returns
        -------
        out: float tensor
        """
        return self.ce(input_, target)


class FocalLoss(nn.Module):
    def __init__(self, num_classes, batch_avg=True, batch_weight=None,
                 class_avg=True, class_weight=None, gamma=2, **kwargs):
        """

        Parameters
        ----------
        num_classes: scalar
            Total number of classes.
        batch_avg:
            Whether to average over the batch dimension.
        batch_weight:
            Batch samples importance coefficients.
        class_avg:
            Whether to average over the class dimension.
        class_weight:
            Classes importance coefficients.
        coeff: scalar
            Gamma coefficient from the paper.
        """
        super().__init__()
        self.num_classes = num_classes
        self.batch_avg = batch_avg
        self.class_avg = class_avg
        self.batch_weight = batch_weight
        self.class_weight = class_weight

        self.gamma = gamma
        logger.warning('Redundant loss function arguments:\n{}'
                       .format(repr(kwargs)))

    def forward(self, input_, target, **kwargs):
        """

        Parameters
        ----------
        input_: (b, ch, d0, d1) tensor
        target: (b, d0, d1) tensor

        Returns
        -------
        out: float tensor
        """
        logpt = -F.cross_entropy(input_, target,
                                 weight=self.class_weight, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean().to('cpu')


dict_losses = {
    'bce_loss': nn.BCEWithLogitsLoss,
    'multi_ce_loss': CrossEntropyLoss,
    'multi_focal_loss': FocalLoss,
}
