import numpy as np
import pytest

import counter_attack.distance_tools as distance_tools

def test_lp_distance_correct_distance():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[2, 2, 1], [3, 2, 6]])

    #L_0: Count of different values
    l0 = distance_tools.LpDistanceMeasure(0, False)
    assert l0.compute(x, y, False, (0, 1)) == 4

    #L_1: Sum of absolute differences
    l1 = distance_tools.LpDistanceMeasure(1, False)
    assert l1.compute(x, y, False, (0, 1)) == 7

    #L_2: Square root of the sum of squared differences
    l2 = distance_tools.LpDistanceMeasure(2, False)
    assert l2.compute(x, y, False, (0, 1)) == np.sqrt(15)

    #L_infinity: Maximum difference
    linf = distance_tools.LpDistanceMeasure(np.inf, False)
    assert linf.compute(x, y, False, (0, 1)) == 3