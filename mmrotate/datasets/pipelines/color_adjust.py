# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from numpy import random

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class BrightnessAdjust(object):
    """Adjust brightness of image by adding a random delta to all pixels.

    Args:
        delta (int): Maximum absolute value of the random delta added to
            all pixels. The delta is sampled uniformly from [-delta, delta].
            Default: 24.
        prob (float): Probability of applying the adjustment. Default: 0.5.
    """

    def __init__(self, delta=24, prob=0.5):
        self.delta = delta
        self.prob = prob

    def __call__(self, results):
        if random.rand() > self.prob:
            return results

        delta = random.uniform(-self.delta, self.delta)
        img_fields = results.get('img_fields', ['img'])
        for key in img_fields:
            img = results[key].astype(np.float32)
            img += delta
            img = np.clip(img, 0, 255).astype(np.uint8)
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(delta={self.delta}, prob={self.prob})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class ContrastAdjust(object):
    """Adjust contrast of image by multiplying pixels by a random alpha.

    Args:
        alpha_range (tuple[float]): Range for the random multiplier.
            Sampled uniformly from [alpha_range[0], alpha_range[1]].
            Default: (0.7, 1.3).
        prob (float): Probability of applying the adjustment. Default: 0.5.
    """

    def __init__(self, alpha_range=(0.7, 1.3), prob=0.5):
        self.alpha_range = alpha_range
        self.prob = prob

    def __call__(self, results):
        if random.rand() > self.prob:
            return results

        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
        img_fields = results.get('img_fields', ['img'])
        for key in img_fields:
            img = results[key].astype(np.float32)
            img *= alpha
            img = np.clip(img, 0, 255).astype(np.uint8)
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(alpha_range={self.alpha_range}, prob={self.prob})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class SaturationAdjust(object):
    """Adjust saturation of image by scaling the S channel in HSV space.

    Converts BGR to HSV, multiplies the Saturation (S) channel by a random
    alpha, then converts back to BGR.

    Args:
        alpha_range (tuple[float]): Range for the random multiplier on the
            S channel. Sampled uniformly from [alpha_range[0], alpha_range[1]].
            Default: (0.7, 1.3).
        prob (float): Probability of applying the adjustment. Default: 0.5.
    """

    def __init__(self, alpha_range=(0.7, 1.3), prob=0.5):
        self.alpha_range = alpha_range
        self.prob = prob

    def __call__(self, results):
        if random.rand() > self.prob:
            return results

        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
        img_fields = results.get('img_fields', ['img'])
        for key in img_fields:
            img = results[key]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[..., 1] *= alpha
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            hsv = hsv.astype(np.uint8)
            results[key] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(alpha_range={self.alpha_range}, prob={self.prob})'
        return repr_str
