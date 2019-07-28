import logging
from progress.bar import IncrementalBar

import numpy as np

from counter_attack import utils

logger = logging.getLogger(__name__)

def get_iterator(name, logger, loader):
    if logger.getEffectiveLevel() == logging.INFO:
        return IncrementalBar(name).iter(loader)
    else:
        return loader

# https://en.wikipedia.org/wiki/N-sphere
def _l2_ball(radius, shape):
    count = np.prod(shape)
    random_direction = utils.random_unit_vector([count])
    u = np.random.random()
    
    random_point = random_direction * np.power(u, 1 / count)
    random_point = random_point * radius

    assert utils.lp_norm(random_point, 2) <= radius
    logger.debug('Generated L2 point with norm {} (radius: {})'.format(utils.lp_norm(random_point, 2), radius))

    return random_point.reshape(shape)

# Just an hypercube
def _linf_ball(radius, shape):
    random_point = (np.random.rand(*shape) * 2 - 1) * radius

    logger.debug('Generated LInf point with norm {} (radius: {})'.format(utils.lp_norm(random_point, np.inf), radius))

    return random_point


def sample_lp_ball(p, radius, shape):
    if p == 2:
        return _l2_ball(radius, shape)
    elif np.isposinf(p):
        return _linf_ball(radius, shape)
    else:
        return NotImplementedError('Ball sampling is only supported for L2 and LInf norms.')
