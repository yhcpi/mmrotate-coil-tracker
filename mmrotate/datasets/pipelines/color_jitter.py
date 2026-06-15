# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from numpy import random

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class ColorJitter(object):
    """Apply random brightness, contrast, saturation and hue to image.

    Similar to torchvision.transforms.ColorJitter but works with numpy BGR
    images and cv2. Each enabled adjustment (brightness/contrast/saturation/
    hue) is applied independently with 50% probability, and the order of
    the enabled adjustments is randomized.

    Args:
        brightness (float | tuple[float]): How much to jitter brightness.
            If float, brightness is sampled from [max(0, 1-b), 1+b].
            If tuple, sampled from [brightness[0], brightness[1]].
            If 0 or (0,0), brightness adjustment is disabled. Default: 0.
        contrast (float | tuple[float]): How much to jitter contrast.
            Same semantics as brightness. Default: 0.
        saturation (float | tuple[float]): How much to jitter saturation.
            Same semantics as brightness. Default: 0.
        hue (float | tuple[float]): How much to jitter hue.
            If float, hue is sampled from [-h, h], clamped to [-0.5, 0.5].
            If tuple, from [hue[0], hue[1]].
            If 0 or (0,0), hue adjustment is disabled. Default: 0.
        prob (float): Probability of applying ColorJitter. Default: 0.5.
    """

    def __init__(self,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 prob=0.5):
        self.brightness = self._check_param(brightness, 'brightness')
        self.contrast = self._check_param(contrast, 'contrast')
        self.saturation = self._check_param(saturation, 'saturation')
        self.hue = self._check_hue(hue)
        self.prob = prob

    @staticmethod
    def _check_param(val, name):
        """Normalize param to (low, high) tuple, or (0, 0) if disabled."""
        if isinstance(val, (int, float)):
            if val == 0:
                return (0.0, 0.0)
            low = max(0, 1 - val)
            high = 1 + val
            return (low, high)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            if val[0] == 0 and val[1] == 0:
                return (0.0, 0.0)
            return (float(val[0]), float(val[1]))
        raise TypeError(
            f'{name} must be a float or (low, high) tuple, got {type(val)}')

    @staticmethod
    def _check_hue(val):
        """Normalize hue param and clamp to [-0.5, 0.5]."""
        if isinstance(val, (int, float)):
            if val == 0:
                return (0.0, 0.0)
            val = max(0, min(val, 0.5))
            return (-val, val)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            low, high = float(val[0]), float(val[1])
            low = max(low, -0.5)
            high = min(high, 0.5)
            return (low, high)
        raise TypeError(f'hue must be a float or (low, high) tuple, got {type(val)}')

    def _adjust_brightness(self, img, delta):
        img = img.astype(np.float32)
        img += delta
        return np.clip(img, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, img, alpha):
        img = img.astype(np.float32)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (1 - alpha) * mean + alpha * img
        return np.clip(img, 0, 255).astype(np.uint8)

    def _adjust_saturation(self, img, alpha):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= alpha
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _adjust_hue(self, img, delta):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Hue in OpenCV HSV is [0, 179] (maps to [0, 360) degrees)
        # delta is normalized to [-0.5, 0.5] → convert to OpenCV units [0, 179]
        delta_opencv = delta * 179
        hsv[..., 0] += delta_opencv
        hsv[..., 0] = np.mod(hsv[..., 0], 180)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def __call__(self, results):
        if random.rand() > self.prob:
            return results

        enabled_fns = []
        if self.brightness != (0.0, 0.0):
            enabled_fns.append(('brightness', self.brightness))
        if self.contrast != (0.0, 0.0):
            enabled_fns.append(('contrast', self.contrast))
        if self.saturation != (0.0, 0.0):
            enabled_fns.append(('saturation', self.saturation))
        if self.hue != (0.0, 0.0):
            enabled_fns.append(('hue', self.hue))

        if not enabled_fns:
            return results

        # Randomize the order the transforms are applied in
        random.shuffle(enabled_fns)

        img_fields = results.get('img_fields', ['img'])
        for key in img_fields:
            img = results[key].copy()
            for name, param in enabled_fns:
                if name == 'brightness':
                    delta = random.uniform(-param[1], param[1])
                    img = self._adjust_brightness(img, delta)
                elif name == 'contrast':
                    alpha = random.uniform(param[0], param[1])
                    img = self._adjust_contrast(img, alpha)
                elif name == 'saturation':
                    alpha = random.uniform(param[0], param[1])
                    img = self._adjust_saturation(img, alpha)
                elif name == 'hue':
                    delta = random.uniform(param[0], param[1])
                    img = self._adjust_hue(img, delta)
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation}, hue={self.hue}, '
        repr_str += f'prob={self.prob})'
        return repr_str
