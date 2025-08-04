import numpy as np


def circular_convolve(arrivals: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular discrete convolution of ``arrivals`` with ``kernel``.

    Parameters
    ----------
    arrivals : np.ndarray
        1‑D array where each element is the number of vehicles starting to
        charge at the corresponding time slot.
    kernel : np.ndarray
        1‑D array describing the power contribution of a single vehicle over
        subsequent slots.

    Returns
    -------
    np.ndarray
        Array of the same length as ``arrivals`` containing the aggregated
        power across all slots.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    n = arrivals.size
    result = np.zeros(n, dtype=float)
    for start, count in enumerate(arrivals):
        if count == 0:
            continue
        for k, val in enumerate(kernel):
            result[(start + k) % n] += count * val
    return result


def aggregate_power(arrivals: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Aggregate charging power for multiple vehicle categories.

    Parameters
    ----------
    arrivals : np.ndarray
        2‑D array with shape (n_slots, n_types) where each column contains the
        number of vehicles of a given type starting to charge in each slot.
    kernels : np.ndarray
        2‑D array with shape (kernel_len, n_types). Column ``t`` represents the
        per‑slot power profile of vehicles of type ``t``.

    Returns
    -------
    np.ndarray
        1‑D array of length ``n_slots`` giving the total power demand in each
        slot after convolving arrivals with their respective kernels.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    n_slots, n_types = arrivals.shape
    total = np.zeros(n_slots, dtype=float)
    for t in range(n_types):
        total += circular_convolve(arrivals[:, t], kernels[:, t])
    return total

