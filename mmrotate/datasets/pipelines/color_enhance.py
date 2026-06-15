# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from numpy import random

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class Sharpness(object):
    """Apply unsharp-mask sharpening/blurring to image.

    Args:
        alpha (tuple[float]): Strength range. alpha > 0 sharpens, alpha < 0
            blurs. Sampled from [alpha[0], alpha[1]]. Default: (0.5, 2.0).
        prob (float): Probability of applying. Default: 0.5.
    """

    def __init__(self, alpha=(0.5, 2.0), prob=0.5):
        assert isinstance(alpha, (list, tuple)) and len(alpha) == 2
        assert alpha[0] <= alpha[1]
        self.alpha = alpha
        self.prob = prob

    def __call__(self, results):
        if random.rand() > self.prob:
            return results

        alpha = random.uniform(self.alpha[0], self.alpha[1])

        img_fields = results.get('img_fields', ['img'])
        for key in img_fields:
            img = results[key]
            blurred = cv2.GaussianBlur(img, (0, 0), 1.5)
            sharp = img.astype(np.float32) + (img.astype(np.float32) - blurred.astype(np.float32)) * alpha
            results[key] = np.clip(sharp, 0, 255).astype(np.uint8)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(alpha={self.alpha}, prob={self.prob})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class ColorTemperature(object):
    """Adjust color temperature (warm/cool) of image.

    Warm tones: shift toward red/yellow (increase R, decrease B).
    Cool tones: shift toward blue (increase B, decrease R).

    Args:
        delta (float): Maximum strength. The B and R channels are scaled by
            factors (1 +/- delta) and (1 -/+ delta) respectively.
            e.g. delta=0.1 means B * 0.9 and R * 1.1 for warm, or vice versa.
            Sampled from [-delta, delta] so both warm and cool are possible.
            Default: 0.15.
        prob (float): Probability of applying. Default: 0.5.
    """

    def __init__(self, delta=0.15, prob=0.5):
        self.delta = delta
        self.prob = prob

    def __call__(self, results):
        if random.rand() > self.prob:
            return results

        t = random.uniform(-self.delta, self.delta)

        img_fields = results.get('img_fields', ['img'])
        for key in img_fields:
            img = results[key].astype(np.float32)
            # BGR: B *= (1 + t), G unchanged, R *= (1 - t)
            # t > 0: B up (cooler), R down
            # t < 0: B down (warmer), R up
            img[..., 0] *= (1.0 + t)
            img[..., 2] *= (1.0 - t)
            results[key] = np.clip(img, 0, 255).astype(np.uint8)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(delta={self.delta}, prob={self.prob})'
        return repr_str
