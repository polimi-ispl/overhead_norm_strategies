"""
Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Nicolò Bonettini - nicolo.bonettini@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
"""

# Libraries import #
import keras
import numpy as np
from keras import backend as K
import tensorflow as tf


# --- Fingerprint extractor losses --- #

class DistanceBasedLogisticLoss(object):
    """
    Class implementing the distance based logistic (DBL) loss (https://arxiv.org/abs/1608.00161)
    """

    def __init__(self, batch_size: int, num_pos: int, num_imgs: int):
        """
        Constructor method
        :param batch_size: int, number of images in the batch
        :param num_pos: int, number of position considered for the patches extraction
        :param num_imgs: int, number of tiles per acquisition considered
        """
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.num_imgs = num_imgs
        self.tot_patch = batch_size*num_pos*num_imgs
        self.__name__ = 'dbl'

    def mse_patch_dbl(self, y_true, y_pred):
        """
        Implementation of the DBL loss using MSE as a distance metric.
        For each patch processed in the batch, we computed the pairwise MSE and save them in a matrix.
        Then we create a corresponding matrix of labels: if the patches come from the same acquisition, the label is
        positive (+1), otherwise negative.

        :param y_true: batch labels for the loss
        :param y_pred: batch predictions
        :return: mse logistic loss value
        """

        # Flatten the prediction in an array of # of total patches in the batch
        y_pred_flat = tf.reshape(y_pred, [self.tot_patch, -1])

        # Compute the pairwise MSE
        A = y_pred_flat
        B = tf.transpose(A)  # needed for TF efficient computation

        na = tf.tile(tf.expand_dims(tf.reduce_sum(tf.square(A), axis=1), -1), [1, self.tot_patch])
        nb = tf.tile(tf.expand_dims(tf.reduce_sum(tf.square(B), axis=0), 0), [self.tot_patch, 1])

        l2norm = na - 2. * tf.matmul(A, B) + nb  # efficient MSE computation

        # Normalize the distances to compute the logistic loss
        # first force the values on the diagonal of the distances matrix to be = inf: this is needed since
        # on the diagonal we will have the distance of the patch w/ itself, making the loss unstable
        l2norm = tf.where(tf.eye(tf.shape(l2norm)[0]) > 0, np.inf * tf.ones_like(l2norm), l2norm)
        l2norm_softmax = tf.exp(-l2norm) / \
                         tf.tile(tf.expand_dims(tf.reduce_sum(tf.exp(-l2norm), axis=1) + K.epsilon(), -1),
                                 [1, self.tot_patch])  # compute Softmax (i.e., normalize between 0-1 the distances)

        # Compute the logistic loss (vectorized form for efficiency)
        l2norm_vec = tf.reshape(l2norm_softmax, [-1])  # vectorize the normalized distances
        y_true_vec = tf.reshape(y_true, [-1])  # vectorize the labels
        check_labels = l2norm_vec * y_true_vec  # compute the argument for the log function
        dbl = tf.reduce_sum(-tf.math.log(tf.reduce_sum(tf.reshape(check_labels,
                                                                  [self.tot_patch, self.tot_patch]),
                                                       axis=1) + K.epsilon()))

        return dbl

    def __call__(self, y_true, y_pred):
        """
        Callable function for working with the Keras API, parameters are the same as the mse_patch_dbl function
        :param y_true: batch labels for the loss
        :param y_pred: batch predictions
        :return: dbl loss computation
        """
        return self.mse_patch_dbl(y_true, y_pred)


# --- Unet segmentation losses --- #

SMOOTH = 1e-5

# ----------------------------------------------------------------
#   Helpers and utilities
# ----------------------------------------------------------------


def _gather_channels(x, indexes):
    """Slice tensor along channels axis by given indexes"""
    if K.image_data_format() == 'channels_last':
        x = K.permute_dimensions(x, (3, 0, 1, 2))
        x = K.gather(x, indexes)
        x = K.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = K.permute_dimensions(x, (1, 0, 2, 3))
        x = K.gather(x, indexes)
        x = K.permute_dimensions(x, (1, 0, 2, 3))
    return x


def get_reduce_axes(per_image):
    axes = [1, 2] if K.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x


def average(x, per_image=False, class_weights=None):
    if per_image:
        x = K.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)


# ----------------------------------------------------------------
#   Metric Functions
# ----------------------------------------------------------------

def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:
    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
    Returns:
        IoU/Jaccard score in range [0, 1]
    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index
    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # score calculation
    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights)

    return score


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:
    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}
    The formula in terms of *Type I* and *Type II* errors:
    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}
    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
    Returns:
        F-score in range [0, 1]
    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # calculate score
    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights)

    return score


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r"""Calculate precision between the ground truth (gt) and the prediction (pr).
    .. math:: F_\beta(tp, fp) = \frac{tp} {(tp + fp)}
    where:
         - tp - true positives;
         - fp - false positives;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.
    Returns:
        float: precision score
    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # score calculation
    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp

    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, per_image, class_weights)

    return score


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r"""Calculate recall between the ground truth (gt) and the prediction (pr).
    .. math:: F_\beta(tp, fn) = \frac{tp} {(tp + fn)}
    where:
         - tp - true positives;
         - fp - false positives;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.
    Returns:
        float: recall score
    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = K.sum(gt * pr, axis=axes)
    fn = K.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights)

    return score


# ----------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------

def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None):

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    # scale predictions so that the class probas of each sample sum to 1
    axis = 3 if K.image_data_format() == 'channels_last' else 1
    pr /= K.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's
    pr = K.clip(pr, K.epsilon(), 1 - K.epsilon())

    # calculate loss
    output = gt * K.log(pr) * class_weights
    return - K.mean(output)


def binary_crossentropy(gt, pr):
    return K.mean(K.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None):
    r"""Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    # clip to prevent NaN's and Inf's
    pr = K.clip(pr, K.epsilon(), 1.0 - K.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr))

    return K.mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25):
    r"""Implementation of Focal Loss from the paper in binary classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
               - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
    """

    # clip to prevent NaN's and Inf's
    pr = K.clip(pr, K.epsilon(), 1.0 - K.epsilon())

    loss_1 = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * K.pow((pr), gamma) * K.log(1 - pr))
    loss = K.mean(loss_0 + loss_1)
    return loss


# ----------------------------------------------------------------
#   Loss and Metrics classes
# ----------------------------------------------------------------

class Loss(object):
    def __init__(self, name):
        self.__name__ = name

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{}({})'.format(multiplier, loss.__name__)
        else:
            name = '{}{}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, gt, pr):
        return self.multiplier * self.loss(gt, pr)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{}_plus_{}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, gt, pr):
        return self.l1(gt, pr) + self.l2(gt, pr)


class DiceLoss(Loss):
    r"""Creates a criterion to measure Dice loss:
    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}
    The formula in terms of *Type I* and *Type II* errors:
    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot fp + \beta^2 \cdot fn + fp}
    where:
         - tp - true positives;
         - fp - false positives;
         - fn - false negatives;
    Args:
        beta: Float or integer coefficient for precision and recall balance.
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged,
        else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.
    Returns:
        A callable ``dice_loss`` instance. Can be used in ``model.compile(...)`` function`
        or combined with other losses.
    Example:
    .. code:: python
        loss = DiceLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return 1 - f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
        )


class BinaryFocalLoss(Loss):
    r"""Creates a criterion that measures the Binary Focal Loss between the
    ground truth (gt) and the prediction (pr).
    .. math:: L(gt, pr) = - gt \alpha (1 - pr)^\gamma \log(pr) - (1 - gt) \alpha pr^\gamma \log(1 - pr)
    Args:
        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.
        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.
    Returns:
        A callable ``binary_focal_loss`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    Example:
    .. code:: python
        loss = BinaryFocalLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, gt, pr):
        return binary_focal_loss(gt, pr, alpha=self.alpha, gamma=self.gamma)


class IOUScore(object):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:
    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}
    Args:
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
    Returns:
       A callable ``iou_score`` instance. Can be used in ``model.compile(...)`` function.
    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index
    Example:
    .. code:: python
        metric = IOUScore()
        model.compile('SGD', loss=loss, metrics=[metric])
    """

    def __init__(
            self,
            class_weights=None,
            class_indexes=None,
            threshold=None,
            per_image=False,
            smooth=SMOOTH,
            name=None,
    ):
        name = name or 'iou_score'
        self.name = name
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
        )


class FScore(object):
    """The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:
    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}
    The formula in terms of *Type I* and *Type II* errors:
    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot fp + \beta^2 \cdot fn + fp}
    where:
         - tp - true positives;
         - fp - false positives;
         - fn - false negatives;
    Args:
        beta: Integer of float f-score coefficient to balance precision and recall.
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``f{beta}-score`` name is used.
    Returns:
        A callable ``f_score`` instance. Can be used in ``model.compile(...)`` function.
    Example:
    .. code:: python
        metric = FScore()
        model.compile('SGD', loss=loss, metrics=[metric])
    """

    def __init__(
            self,
            beta=1,
            class_weights=None,
            class_indexes=None,
            threshold=None,
            per_image=False,
            smooth=SMOOTH,
            name=None,
    ):
        name = name or 'f{}-score'.format(beta)
        self.name = name
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
        )