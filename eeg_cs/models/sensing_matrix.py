from abc import ABC

import numpy as np
import numpy.typing as npt


class SensingMatrix(ABC):
    """
    Abstract class for sensing matrices.
    """

    _value: npt.NDArray[np.float64]

    @property
    def value(self) -> npt.NDArray[np.float64]:
        return self._value

class Gaussian(SensingMatrix):
    """
    Gaussian sensing matrix.
    """

    def __init__(self, m: int, n: int, random_state: int = 42):
        rng = np.random.default_rng(random_state)
        self._value = rng.normal(loc=0.0, scale=1, size=(m, n))

class Bernoulli(SensingMatrix):
    """
    Bernoulli random sensing matrix (+1/-1 entries).
    """

    def __init__(self, m: int, n: int, random_state: int = 42):
        rng = np.random.default_rng(random_state)
        self._value = rng.choice([-1, 1], size=(m, n))


class Undersampled(SensingMatrix):
    """
    Undersampled identity matrix.
    """

    def __init__(self, m: int, n: int, random_state: int = 42):
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(n, m, replace=False))
        self._value = np.eye(n)[indices, :]


class BinaryPermutedBlockDiagonal(SensingMatrix):
    """
    Binary permuted block diagonal matrix.
    """

    def __init__(self, M: int, CR: int):
        Identity = np.eye(M, dtype=int)
        self._value = np.repeat(Identity, repeats=CR, axis=1)


class SparseBinary(SensingMatrix):
    """
    Sparse binary matrix with `d` ones in each column in random rows.
    """

    def __init__(self, m: int, n: int, d: int, random_state: int = 42):
        if d > m:
            raise ValueError(
                "The number of ones per column (d) cannot exceed the number of rows (m)."
            )

        rng = np.random.default_rng(random_state)
        self._value = np.zeros((m, n), dtype=int)

        for col in range(n):
            row_indices = rng.choice(m, d, replace=False)
            self._value[row_indices, col] = 1
