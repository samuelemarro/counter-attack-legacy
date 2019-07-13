import numpy as np
import pytest

from counter_attack import distance_measures

def test_lp_distance_correct_distance():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[2, 2, 1], [3, 2, 6]])

    #L_0: Count of different values
    l0 = distance_measures.LpDistance(0)
    assert l0.compute(x, y, False) == 4

    #L_1: Sum of absolute differences
    l1 = distance_measures.LpDistance(1)
    assert l1.compute(x, y, False) == 7

    #L_2: Square root of the sum of squared differences
    l2 = distance_measures.LpDistance(2)
    assert l2.compute(x, y, False) == np.sqrt(15)

    #L_infinity: Maximum difference
    linf = distance_measures.LpDistance(np.inf)
    assert linf.compute(x, y, False) == 3