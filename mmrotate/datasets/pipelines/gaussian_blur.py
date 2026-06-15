# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from numpy import random

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class GaussianBlur(object):
    """Apply Gaussian blur to image.

    Args:
        kernel_size (int | tuple[int]): Size of the Gaussian kernel.
            If int, the kernel will be (kernel_size, kernel_size).
            Default: (5, 5).
        sigma (float | tuple[float]): Standard deviation of the Gaussian kernel.
            If float, sigma is fixed. If tuple, sigma is randomly sampled
            from (sigma[0], sigma[1]). Default: (0.1, 2.0).
        prob (float): Probability of applying the blur. Default: 0.5.
    """

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), prob=0.5):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, (list, tuple))
        assert len(kernel_size) == 2
        assert all(k % 2 == 1 for k in kernel_size), \
            'kernel_size must be odd numbers'
        self.kernel_size = kernel_size
        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma)
        assert isinstance(sigma, (list, tuple))
        assert len(sigma) == 2
        assert sigma[0] <= sigma[1]
        self.sigma = sigma
        self.prob = prob

    def __call__(self, results):
        """Call function to apply Gaussian blur to image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with Gaussian blur applied.
        """
        if random.rand() > self.prob:
            return results

        if 'img_fields' in results:
            for key in results.get('img_fields', []):
                img = results[key]
                ksize = self.kernel_size
                sigma = random.uniform(self.sigma[0], self.sigma[1])
                results[key] = cv2.GaussianBlur(img, ksize, sigma)
        else:
            img = results['img']
            ksize = self.kernel_size
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            results['img'] = cv2.GaussianBlur(img, ksize, sigma)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(kernel_size={self.kernel_size}, '
        repr_str += f'sigma={self.sigma}, '
        repr_str += f'prob={self.prob})'
        return repr_str
