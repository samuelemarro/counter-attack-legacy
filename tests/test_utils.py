import pytest
#from context import OffenseDefense
import OffenseDefense.utils as utils
import numpy as np

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
    assert utils.lp_distance(x, y, np.Infinity, False) == 3

def test_is_top_k_correct():
    predictions = [0.4, 0.1, 0.9, -0.5, 0.2]
    #Labels, sorted by prediction:
    #2 (0.9)
    #0 (0.4)
    #4 (0.2)
    #1 (0.1)
    #3 (-0.5)

    assert utils.is_top_k(predictions, 2, k=1)
    assert utils.is_top_k(predictions, 4, k=3)
    assert not utils.is_top_k(predictions, 1, k=2)
    assert not utils.is_top_k(predictions, 3, k=1)
