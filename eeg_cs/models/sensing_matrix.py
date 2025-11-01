from abc import ABC

import numpy as np
import numpy.typing as npt


class SensingMatrix(ABC):
  """
  Abstract class for sensing matrices.
  """

  _name: str
  _value: npt.NDArray[np.float64]
  _scale: float
  _value_normalized: npt.NDArray[np.float64]

  @property
  def name(self) -> str:
    return self._name

  @property
  def value(self) -> npt.NDArray[np.float64]:
    return self._value

  @property
  def scale(self) -> float:
    return self._scale

  @property
  def value_normalized(self) -> npt.NDArray[np.float64]:
    return self._value_normalized


class Gaussian(SensingMatrix):
  """
  Gaussian sensing matrix.
  Standard normalization: each row has unit norm (scale by 1/√n).
  """

  def __init__(self, m: int, n: int, random_state: int = 42) -> None:
    self._name = "Gaussian"

    rng = np.random.default_rng(random_state)
    self._value = rng.normal(loc=0.0, scale=1, size=(m, n))
    # Standard CS normalization: scale by 1/√n for unit row norm expectation
    self._scale = 1.0 / np.sqrt(n)
    self._value_normalized = self.value * self.scale


class Bernoulli(SensingMatrix):
  """
  Bernoulli random sensing matrix (+1/-1 entries).
  Standard normalization: each row has unit norm (scale by 1/√n).
  """

  def __init__(self, m: int, n: int, random_state: int = 42) -> None:
    self._name = "Bernoulli"

    rng = np.random.default_rng(random_state)
    self._value = rng.choice([-1, 1], size=(m, n))
    # Standard CS normalization: scale by 1/√n for unit row norm
    self._scale = 1.0 / np.sqrt(n)
    self._value_normalized = self.value * self.scale


class Undersampled(SensingMatrix):
  """
  Undersampled identity matrix.
  No normalization needed as each row already has unit norm.
  """

  def __init__(self, m: int, n: int, random_state: int = 42) -> None:
    self._name = "Undersampled"

    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(n, m, replace=False))
    self._value = np.eye(n)[indices, :]
    # No normalization needed - each row of identity matrix has unit norm
    self._scale = 1.0
    self._value_normalized = self.value * self.scale


class BinaryPermutedBlockDiagonal(SensingMatrix):
  """
  Binary permuted block diagonal matrix.
  Normalize by 1/√CR to account for repeated identity blocks.
  """

  def __init__(self, M: int, CR: int) -> None:
    self._name = "BPBD"

    identity = np.eye(M, dtype=int)
    self._value = np.repeat(identity, repeats=CR, axis=1)
    # Normalize by 1/√CR since each row has norm √CR due to CR repetitions
    self._scale = 1.0 / np.sqrt(CR)
    self._value_normalized = self.value * self.scale


class SparseBinary(SensingMatrix):
  """
  Sparse binary matrix with `d` ones in each column in random rows.
  Each row has approximately d*n/m ones, so normalize by 1/√(d*n/m).
  """

  def __init__(self, m: int, n: int, d: int, random_state: int = 42) -> None:
    if d > m:
      error_msg = (
        "The number of ones per column (d) cannot exceed the number of rows (m)."
      )
      raise ValueError(error_msg)

    self._name = f"SB_{d}"

    rng = np.random.default_rng(random_state)
    self._value = np.zeros((m, n), dtype=int)

    for col in range(n):
      row_indices = rng.choice(m, d, replace=False)
      self._value[row_indices, col] = 1

    # Each row has approximately d*n/m ones, so normalize by 1/√(d*n/m)
    expected_ones_per_row = d * n / m
    self._scale = 1.0 / np.sqrt(expected_ones_per_row)
    self._value_normalized = self.value * self.scale
