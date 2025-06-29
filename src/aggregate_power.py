import numpy as np


def circular_convolve(arrivals: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution of arrivals with a kernel.

    Parameters
    ----------
    arrivals : array_like of shape (P,)
        Number of cars arriving in each slot.
    kernel : array_like of shape (L,)
        Charging power profile for a single car.

    Returns
    -------
    np.ndarray of shape (P,)
        Power demand contributed by this vehicle type.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    P = arrivals.size
    L = kernel.size
    result = np.zeros(P, dtype=float)
    for i in range(P):
        if arrivals[i] == 0:
            continue
        for k in range(L):
            idx = (i + k) % P
            result[idx] += arrivals[i] * kernel[k]
    return result


def aggregate_power(arrivals: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Aggregate charging power over all vehicle types.

    Parameters
    ----------
    arrivals : array_like of shape (P, T)
        Arrivals per time slot for each vehicle type.
    kernels : array_like of shape (L, T)
        Charging power profiles for each vehicle type.

    Returns
    -------
    np.ndarray of shape (P,)
        Total charging power across all types.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    P, T = arrivals.shape
    L, T2 = kernels.shape
    if T2 != T:
        raise ValueError("arrivals and kernels must have the same number of vehicle types")

    power_by_type = np.zeros((P, T), dtype=float)
    for j in range(T):
        power_by_type[:, j] = circular_convolve(arrivals[:, j], kernels[:, j])
    return power_by_type.sum(axis=1)
