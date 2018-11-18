import pytest
from context import OffenseDefense
from OffenseDefense import utils
import numpy as np

def test_lp_distance_correct_distance():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[2, 2, 1], [3, 2, 6]])

    print(dir(OffenseDefense))

    #L_0: Count of different values
    assert utils.lp_distance(x, y, 0, False) == 4
    #L_2: Square root of the sum of squared differences
    assert utils.lp_distance(x, y, 2, False) == np.sqrt(15)
    #L_infinity: Maximum difference
    assert utils.lp_distance(x, y, np.Infinity, False) == 3

