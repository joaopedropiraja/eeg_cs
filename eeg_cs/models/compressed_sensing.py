from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .reconstruction_algorithm import ReconstructionAlgorithm
from .sensing_matrix import SensingMatrix
from .sparsifying_matrix import SparsifyingMatrix


@dataclass
class CompressedSensing:
  sensing_matrix: SensingMatrix
  sparse_bases: SparsifyingMatrix
  reconstruction_algorithm: ReconstructionAlgorithm
  center_data: bool = True

  _X_mean: npt.NDArray[np.float64] | None = field(init=False, default=None)
  Theta: npt.NDArray[np.float64] = field(init=False)

  def __post_init__(self) -> None:
    self.Theta = self.sparse_bases.apply_sensing_matrix(self.sensing_matrix.value)

  def compress(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    Y = self.sensing_matrix.value @ self._preprocess(X)

    return Y

  def reconstruct(self, Y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    S = self.reconstruction_algorithm.solve(Y, self.Theta)
    X_hat = self.sparse_bases.transform(S)

    return self._postprocess(X_hat)

  @classmethod
  def evaluate(
    cls, X: npt.NDArray[np.float64], X_hat: npt.NDArray[np.float64]
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate metrics for the reconstruction.
    """

    prd = cls._pdr(X, X_hat)
    nmse = cls._nmse(X, X_hat)
    sndr = cls._sndr(X, X_hat)

    return prd, nmse, sndr

  @staticmethod
  def _pdr(
    X: npt.NDArray[np.float64], X_hat: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Calculate percentage of distortion reduction (PDR).
    """
    mu = X.mean(axis=0)

    num = np.linalg.norm(X - X_hat, axis=0) ** 2
    den = np.linalg.norm(X - mu, axis=0) ** 2

    return np.where(den == 0, np.inf, 100 * (num / den))

  @staticmethod
  def _nmse(
    X: npt.NDArray[np.float64], X_hat: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Calculate normalized mean square error (NMSE).
    """

    mu = X.mean(axis=0)

    num = np.linalg.norm(X - X_hat, axis=0) ** 2
    den = np.linalg.norm(X - mu, axis=0) ** 2

    return np.where(den == 0, np.inf, num / den)

  @staticmethod
  def _sndr(
    X: npt.NDArray[np.float64], X_hat: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Calculate signal to noise and distortion ratio (SNDR).
    """

    num = np.linalg.norm(X_hat, axis=0) ** 2
    den = np.linalg.norm(X_hat - X, axis=0) ** 2

    return np.where(den == 0, np.inf, 10 * np.log10(num / den))

  def _preprocess(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if self.center_data:
      self._X_mean = X.mean(axis=0, keepdims=True)
      return X - self._X_mean

    return X

  def _postprocess(self, X_hat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if self.center_data and self._X_mean is not None:
      return X_hat + self._X_mean

    return X_hat
