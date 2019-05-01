import pytest
#from context import OffenseDefense
import OffenseDefense.utils as utils
import numpy as np

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
