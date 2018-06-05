import logging

import torch
import numpy as np


logging.basicConfig()
logger = logging.getLogger('metrics')
logger.setLevel(logging.DEBUG)


def confusion_matrix(input_, target, num_classes):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py

    Args:
        input_: (d0, ..., dn) ndarray or tensor
        target: (d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (num_classes, num_classes) ndarray
            Confusion matrix.
    """
    if torch.is_tensor(input_):
        input_ = input_.detach().to('cpu').numpy()
    if torch.is_tensor(target):
        target = target.detach().to('cpu').numpy()

    replace_indices = np.vstack((
        target.flatten(),
        input_.flatten())
    ).T
    cm, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes-1), (0, num_classes-1)]
    )
    return cm.astype(np.uint32)


def dice_score_from_cm(cm):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py

    Args:
        cm: (d, d) ndarray
            Confusion matrix.
    
    Returns:
        out: (d, ) list
            List of class Dice scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = 2 * float(true_positives) / denom
        scores.append(score)
    return scores


def jaccard_score_from_cm(cm):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py

    Args:
        cm: (d, d) ndarray
            Confusion matrix.
    
    Returns:
        out: (d, ) list
            List of class IoU scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def precision_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class precision scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        denom = true_positives + false_positives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def recall_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.
    
    Returns:
        out: (d, ) list
            List of class recall scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_negatives = cm[index, :].sum() - true_positives
        denom = true_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def sensitivity_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class sensitivity scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_negatives = cm[index, :].sum() - true_positives
        denom = true_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def specificity_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class sensitivity scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        true_negatives = np.trace(cm) - true_positives
        false_positives = cm[:, index].sum() - true_positives
        denom = false_positives + true_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_negatives) / denom
        scores.append(score)
    return scores


def vol_sim_from_cm(cm):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class volumetric similarity scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = 1 - abs(false_negatives - false_positives) / denom
        scores.append(score)
    return scores


# ----------------------------------------------------------------------------


def _template_score(func_score_from_cm, input_, target, num_classes,
                    batch_avg, batch_weight, class_avg, class_weight):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    scores = np.zeros((num_samples, num_classes))
    for sample_idx in range(num_samples):
        cm = confusion_matrix(input_=input_[sample_idx],
                              target=target[sample_idx],
                              num_classes=num_classes)
        scores[sample_idx, :] = func_score_from_cm(cm)

    if batch_avg:
        scores = np.mean(scores, axis=0, keepdims=True)
    if class_avg:
        if class_weight is not None:
            scores = scores * np.reshape(class_weight, (1, -1))
        scores = np.mean(scores, axis=1, keepdims=True)
    return np.squeeze(scores)


def dice_score(input_, target, num_classes,
               batch_avg=True, batch_weight=None,
               class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        dice_score_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def jaccard_score(input_, target, num_classes,
                  batch_avg=True, batch_weight=None,
                  class_avg=False, class_weight=None):
    """Jaccard similarity score, also known as Intersection-over-Union (IoU).

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        jaccard_score_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def precision_score(input_, target, num_classes,
                    batch_avg=True, batch_weight=None,
                    class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        precision_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def recall_score(input_, target, num_classes,
                 batch_avg=True, batch_weight=None,
                 class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        recall_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def sensitivity_score(input_, target, num_classes,
                      batch_avg=True, batch_weight=None,
                      class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        sensitivity_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def specificity_score(input_, target, num_classes,
                      batch_avg=True, batch_weight=None,
                      class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        specificity_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def volumetric_similarity(input_, target, num_classes,
                          batch_avg=True, batch_weight=None,
                          class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        vol_sim_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def volume_error(input_, target, num_classes,
                 batch_avg=True, batch_weight=None,
                 class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """

    def _ve(i, t):
        numer = 2 * np.count_nonzero(np.bitwise_xor(i, t))
        denom = np.count_nonzero(i) + np.count_nonzero(t)
        if denom == 0:
            return 0
        else:
            return numer / denom

    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    scores = np.zeros((num_samples, num_classes))
    for sample_idx in range(num_samples):
        for class_idx in range(num_classes):
            sel_input_ = input_[sample_idx] == class_idx
            sel_target = target[sample_idx] == class_idx

            scores[sample_idx, class_idx] = _ve(sel_input_, sel_target)

    if batch_avg:
        scores = np.mean(scores, axis=0, keepdims=True)
    if class_avg:
        if class_weight is not None:
            scores = scores * np.reshape(class_weight, (1, -1))
        scores = np.mean(scores, axis=1, keepdims=True)
    return np.squeeze(scores)


def volume_total(input_, num_classes, spacing_mm=(1, 1, 1)):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        spacing_mm: 3-tuple
            Pixel spacing in mm, one per each spatial dimension of `input_`.

    Returns:
        out: (b, num_classes) ndarray
            Total volume for each class in each batch sample.
    """
    def _vt(i, s):
        n = np.count_nonzero(i)
        vox = np.prod(s)
        # Naive approach
        return vox * n

    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    scores = np.zeros((num_samples, num_classes))
    for sample_idx in range(num_samples):
        for class_idx in range(num_classes):
            sel_input_ = input_[sample_idx] == class_idx

            scores[sample_idx, class_idx] = _vt(sel_input_, spacing_mm)

    return scores
