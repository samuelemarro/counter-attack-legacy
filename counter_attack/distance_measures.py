from abc import ABC, abstractmethod
import logging

import foolbox
import numpy as np

from . import utils

logger = logging.getLogger(__name__)

class LpDistance:
    def __init__(self, p):
        if p < 0:
            raise ValueError('p must be positive or zero')
            
        self.p = p

    def compute(self, x, y, batch):
        # If x or y are empty, we return an empty array
        empty_x = hasattr(x, '__len__') and len(x) == 0
        empty_y = hasattr(y, '__len__') and len(y) == 0

        if batch and (empty_x or empty_y):
            return np.array([], dtype=np.float)

        def single_image(diff):
            value = utils.lp_norm(diff, self.p)
            
            return value

        if batch:
            return np.array([single_image(_x - _y) for _x, _y in zip(x, y)])
        else:
            return single_image(x - y)

    def __str__(self):
        _name = ''

        _name += 'L-{} Distance'.format(self.p)

        return _name

class FoolboxDistance(foolbox.distances.Distance):
    """Foolbox-compatible distance measure

    If you're wondering what's going on: foolbox attacks accept
    a distance type instead of an actual instance. So, if you want
    to use the MSE distance, you have to pass foolbox.distances.MSE.
    foolbox then calls the type and builds the distance (e.g. MSE(...)).
    This usually works well, but it means that we can't pass any other
    arguments to the distance like the distance measure. We therefore
    use the following wrapper trick: we init and pass the LpDistance instance;
    foolbox will attempt to create the distance by calling distance(...).
    However, since it's an instance, calls are handled by __call__.
    In __call__, we init WrappedLpDistance with the provided arguments
    (in addition to lp_distance) and return it.
    """
    class WrappedDistance(foolbox.distances.Distance):
        def __init__(self, lp_distance, reference=None, other=None, bounds=None, value=None):
            self.lp_distance = lp_distance
            super().__init__(reference, other, bounds, value)

        def _calculate(self):
            assert self.other.shape == self.reference.shape
            value = self.lp_distance.compute(self.other, self.reference, False, self._bounds)

            gradient = None
            return value, gradient

        @property
        def gradient(self):
            raise NotImplementedError

        def name(self):
            return 'Wrapped Distance ({})'.format(self.lp_distance)

    def __init__(self, lp_distance):
        self.lp_distance = lp_distance

    def __call__(self,
                 reference=None,
                 other=None,
                 bounds=None,
                 value=None):
        return FoolboxDistance.WrappedDistance(self.lp_distance, reference, other, bounds, value)

    def _calculate(self):
        raise NotImplementedError()