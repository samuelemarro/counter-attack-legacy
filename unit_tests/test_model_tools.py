import numpy as np
import pytest

from counter_attack import utils

def test_lp_distance_correct_distance():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[2, 2, 1], [3, 2, 6]])

    #L_0: Count of different values
    assert utils.lp_distance(x, y, 0, False) == 4

    #L_1: Sum of absolute differences
    assert utils.lp_distance(x, y, 1, False) == 7

    #L_2: Square root of the sum of squared differences
    assert utils.lp_distance(x, y, 2, False) == np.sqrt(15)

    #L_infinity: Maximum difference
    assert utils.lp_distance(x, y, np.inf, False) == 3