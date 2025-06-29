import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from aggregate_power import aggregate_power, circular_convolve


def test_circular_convolve_basic():
    arrivals = np.array([1, 0, 0, 0])
    kernel = np.array([2, 1])
    result = circular_convolve(arrivals, kernel)
    # one car at slot 0: contributes [2,1]
    expected = np.array([2, 1, 0, 0])
    assert np.allclose(result, expected)


def test_aggregate_power_simple():
    arrivals = np.array([
        [1, 0],  # slot 0: one type0 car
        [0, 1],  # slot 1: one type1 car
        [0, 0],
        [0, 0],
    ])
    kernels = np.array([
        [2, 1],  # type0 kernel [2]; type1 kernel [1]
        [0, 1],  # type0 kernel [0]; type1 kernel [1]
    ])
    total = aggregate_power(arrivals, kernels)
    expected = np.array([2, 1, 1, 0])
    assert np.allclose(total, expected)
