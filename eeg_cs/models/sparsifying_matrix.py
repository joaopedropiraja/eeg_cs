from abc import ABC, abstractmethod

import cr.sparse.lop as crlop
import numpy as np
import numpy.typing as npt
import pywt
from cr.sparse import lop
from scipy.fftpack import dct, dst, idct, idst


class SparsifyingMatrix(ABC):
  """
  Abstract class for sparse basis.
  """

  _value: npt.NDArray[np.float64]
  _name: str

  @property
  def value(self) -> npt.NDArray[np.float64]:
    return self._value

  @property
  def name(self) -> str:
    return self._name

  @abstractmethod
  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]: ...

  @abstractmethod
  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

  def coherence(self, Phi: npt.NDArray[np.float64]) -> float:
    N = Phi.shape[1]
    # 1) normalize each row of Phi
    Phi_norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi_norms[Phi_norms == 0] = 1.0  # Prevent division by zero
    Phi_normed = Phi / Phi_norms
    # 2) normalize each column of Psi
    Psi_norms = np.linalg.norm(self.value, axis=0, keepdims=True)
    Psi_norms[Psi_norms == 0] = 1.0  # Prevent division by zero
    Psi_normed = self.value / Psi_norms
    # 3) form the matrix of all inner products
    G = Phi_normed @ Psi_normed
    # 4) take maximum absolute entry and scale
    return np.sqrt(N) * np.max(np.abs(G))


class DCT(SparsifyingMatrix):
  """
  Discrete Cosine Transform (DCT) basis.
  """

  def __init__(self, N: int) -> None:
    """
    Create a DCT basis of size N.
    """
    # DCT basis matrix for coherence calculation
    self._value = idct(np.eye(N), norm="ortho", axis=0).astype(np.float64)
    self._name = "DCT"

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    # Equivalent to Phi @ self.value
    return dct(Phi, norm="ortho").astype(np.float64)

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return idct(S, norm="ortho", axis=0).astype(np.float64)


class DST(SparsifyingMatrix):
  """
  Discrete Sine Transform (DST) basis.
  """

  def __init__(self, N: int) -> None:
    """
    Create a DST basis of size N.
    """
    # DST basis matrix for coherence calculation
    self._value = idst(np.eye(N), type=2, norm="ortho", axis=0).astype(np.float64)
    self._name = "DST"

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    # Equivalent to Phi @ self.value
    return dst(Phi, type=2, norm="ortho").astype(np.float64)

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return idst(S, type=2, norm="ortho", axis=0).astype(np.float64)


class CS(SparsifyingMatrix):
  """
  Overcomplete basis combining DCT (C) and DST (S).
  """

  def __init__(self, N: int) -> None:
    identity = np.eye(N)
    dct_basis = idct(identity, norm="ortho", axis=0)
    dst_basis = idst(identity, type=2, norm="ortho", axis=0)

    self._value = np.hstack((dct_basis, dst_basis)).astype(np.float64)
    self._name = "CS"

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    return Phi @ self.value

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return self.value @ S


class IDCT(SparsifyingMatrix):
  """
  Overcomplete basis combining Impulse (I) and DCT (C).
  """

  def __init__(self, N: int) -> None:
    identity = np.eye(N)
    dct_basis = idct(identity, norm="ortho", axis=0)

    self._value = np.hstack((identity, dct_basis)).astype(np.float64)
    self._name = "IDCT"

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    return Phi @ self.value

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return self.value @ S


class ICS(SparsifyingMatrix):
  """
  Overcomplete basis combining Impulse (I), DCT (C), and DST (S).
  """

  def __init__(self, N: int) -> None:
    identity = np.eye(N)
    dct_basis = idct(identity, norm="ortho", axis=0)
    dst_basis = idst(identity, type=2, norm="ortho", axis=0)

    self._value = np.hstack((identity, dct_basis, dst_basis)).astype(np.float64)
    self._name = "ICS"

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    return Phi @ self.value

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return self.value @ S


class Wavelet(SparsifyingMatrix):
  """
  Wavelet sparsifying matrix using PyWavelets.
  """

  def __init__(
    self,
    N: int,
    wavelet: str = "db4",
    mode: str = "periodization",
    levels: int | None = None,
  ) -> None:
    """
    Create a wavelet sparsifying matrix.

    Parameters:
    -----------
    N : int
        Signal length (should preferably be power of 2 for orthogonal wavelets)
    wavelet : str
        Wavelet name (e.g., 'db4', 'haar', 'coif2', 'bior2.2', 'dmey')
    mode : str
        Boundary condition mode ('periodization', 'zero', 'symmetric', etc.)
    levels : int, optional
        Number of decomposition levels. If None, uses maximum possible levels.
    """
    self._wavelet = wavelet
    self._mode = mode

    # Determine maximum decomposition levels
    # Calculate max levels manually since pywt.dwt_max_levels might not be available
    if levels is None:
      # Simple calculation for max levels
      self._levels = int(np.log2(N)) - 1 if N > 1 else 1
    else:
      max_levels = int(np.log2(N)) - 1 if N > 1 else 1
      self._levels = min(levels, max_levels)

    # Build the wavelet basis matrix
    self._value = self._build_wavelet_matrix(N)
    self._name = f"Wavelet_{wavelet}_{self._levels}L"

  def _build_wavelet_matrix(self, N: int) -> npt.NDArray[np.float64]:
    """
    Build the wavelet transform matrix by applying DWT to each column of identity matrix.
    """
    # Create identity matrix
    identity = np.eye(N)
    wavelet_matrix = np.zeros((N, N))

    for i in range(N):
      # Apply DWT to each column (basis vector)
      coeffs = pywt.wavedec(
        identity[:, i], self._wavelet, mode=self._mode, level=self._levels
      )
      # Flatten coefficients to create the column of the transform matrix
      coeffs_flat = np.concatenate([np.asarray(c, dtype=np.float64) for c in coeffs])
      # Ensure we have the right size
      if len(coeffs_flat) <= N:
        wavelet_matrix[: len(coeffs_flat), i] = coeffs_flat
      else:
        wavelet_matrix[:, i] = coeffs_flat[:N]

    return wavelet_matrix.astype(np.float64)

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Apply sensing matrix: equivalent to Phi @ Psi where Psi is the wavelet basis.
    For efficient implementation, we use the direct wavelet transform.
    """
    if Phi.ndim == 1:
      Phi = Phi.reshape(1, -1)

    result = np.zeros((Phi.shape[0], self._value.shape[1]))
    for i in range(Phi.shape[0]):
      coeffs = pywt.wavedec(
        Phi[i, :], self._wavelet, mode=self._mode, level=self._levels
      )
      coeffs_flat = np.concatenate([np.asarray(c, dtype=np.float64) for c in coeffs])
      # Ensure we have the right size
      if len(coeffs_flat) <= result.shape[1]:
        result[i, : len(coeffs_flat)] = coeffs_flat
      else:
        result[i, :] = coeffs_flat[: result.shape[1]]

    return result.astype(np.float64)

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Inverse wavelet transform: reconstruct signal from wavelet coefficients.
    """
    if S.ndim == 1:
      S = S.reshape(-1, 1)

    result = np.zeros((S.shape[0], S.shape[1]))

    # Get the coefficient sizes for reconstruction
    temp_coeffs = pywt.wavedec(
      np.zeros(S.shape[0]), self._wavelet, mode=self._mode, level=self._levels
    )
    coeff_slices = []
    start = 0
    for coeff in temp_coeffs:
      coeff_array = np.asarray(coeff, dtype=np.float64)
      end = start + len(coeff_array)
      coeff_slices.append((start, end))
      start = end

    for j in range(S.shape[1]):
      # Reconstruct coefficients structure
      coeffs = []
      for start, end in coeff_slices:
        coeffs.append(S[start:end, j])

      # Perform inverse DWT
      reconstructed = pywt.waverec(coeffs, self._wavelet, mode=self._mode)  # type: ignore
      reconstructed = np.asarray(reconstructed, dtype=np.float64)
      # Ensure correct size
      if len(reconstructed) >= result.shape[0]:
        result[:, j] = reconstructed[: result.shape[0]]
      else:
        result[: len(reconstructed), j] = reconstructed

    return result.astype(np.float64)

  @property
  def wavelet_name(self) -> str:
    return self._wavelet

  @property
  def decomposition_levels(self) -> int:
    return self._levels


class WaveletV2(SparsifyingMatrix):
  """
  Wavelet sparsifying matrix using cr.sparse lop.dwt operators.
  This implementation leverages the high-performance JAX-based DWT operators
  from the cr.sparse library for efficient sparse coding.
  """

  def __init__(
    self,
    N: int,
    wavelet: str = "db4",
    level: int = 1,
  ) -> None:
    """
    Create a wavelet sparsifying matrix using cr.sparse.

    Parameters:
    -----------
    N : int
        Signal length (preferably power of 2 for optimal performance)
    wavelet : str
        Wavelet name supported by cr.sparse (e.g., 'db4', 'haar', 'dmey', 'coif2')
    level : int
        Number of decomposition levels.
    """
    self._wavelet = wavelet
    self._level = level
    self._N = N

    # Create the DWT operators
    # basis=True gives us the synthesis operator (inverse DWT / reconstruction)
    # basis=False gives us the analysis operator (forward DWT / decomposition)
    self._synthesis_op = crlop.dwt(N, wavelet, level=level, basis=True)
    self._analysis_op = crlop.dwt(N, wavelet, level=level, basis=False)

    # For sparsifying matrix, we want the synthesis matrix (Psi)
    # This is the matrix that transforms coefficients back to signal domain
    self._value = lop.to_matrix(self._synthesis_op).astype(np.float64)

    self._name = f"WaveletV2_{wavelet}_{level}L"

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Apply sensing matrix: equivalent to Phi @ Psi where Psi is the wavelet basis.
    This computes the compressed sensing measurement matrix.
    """
    # Simple matrix multiplication: Phi @ Psi
    return np.asarray(Phi @ self.value)

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Transform coefficients back to signal domain.
    This applies the synthesis operator: Psi @ S
    """
    # Apply synthesis operator to transform coefficients to signal
    return np.asarray(self.value @ S)

  @property
  def wavelet_name(self) -> str:
    return self._wavelet

  @property
  def decomposition_levels(self) -> int:
    return self._level

  @property
  def synthesis_operator(self):
    """Access to the underlying cr.sparse synthesis DWT operator"""
    return self._synthesis_op

  @property
  def analysis_operator(self):
    """Access to the underlying cr.sparse analysis DWT operator"""
    return self._analysis_op


# class WaveletV2(SparsifyingMatrix):
#   """
#   Wavelet sparsifying matrix using cr.sparse lop.dwt operators.
#   This implementation leverages the high-performance JAX-based DWT operators
#   from the cr.sparse library for efficient sparse coding.
#   """

#   def __init__(
#     self,
#     N: int,
#     wavelet: str = "db4",
#     level: int | None = None,
#     basis: bool = True,
#   ) -> None:
#     """
#     Create a wavelet sparsifying matrix using cr.sparse.

#     Parameters:
#     -----------
#     N : int
#         Signal length (preferably power of 2 for optimal performance)
#     wavelet : str
#         Wavelet name supported by cr.sparse (e.g., 'db4', 'haar', 'dmey', 'coif2')
#     level : int, optional
#         Number of decomposition levels. If None, uses maximum possible levels.
#     basis : bool
#         If True, returns the wavelet basis operator (inverse transform first).
#         If False, returns the transform operator (forward transform first).
#     """
#     try:
#       # Import cr.sparse components
#       import jax.numpy as jnp
#       from cr.sparse import lop
#     except ImportError:
#       raise ImportError(
#         "cr.sparse library is required for WaveletV2. Install with 'pip install cr-sparse'"
#       )

#     self._wavelet = wavelet
#     self._level = level
#     self._basis = basis
#     self._N = N

#     # Create the DWT operator
#     self._dwt_op = lop.dwt(N, wavelet=wavelet, level=level, basis=basis)
#     self._dwt_op = lop.jit(self._dwt_op)  # JIT compile for performance

#     # Build the matrix representation for coherence calculations
#     # This is computationally expensive but needed for the coherence method
#     identity = np.eye(N)
#     matrix_cols = []

#     for i in range(N):
#       if basis:
#         # For basis mode, apply the operator directly (times = inverse transform)
#         col = np.array(self._dwt_op.times(jnp.array(identity[:, i])))
#       else:
#         # For transform mode, apply transpose (trans = inverse transform)
#         col = np.array(self._dwt_op.trans(jnp.array(identity[:, i])))
#       matrix_cols.append(col)

#     self._value = np.column_stack(matrix_cols).astype(np.float64)
#     self._name = f"WaveletV2_{wavelet}_{level}L_{'basis' if basis else 'transform'}"

#   def apply_sensing_matrix(
#     self, Phi: npt.NDArray[np.float64]
#   ) -> npt.NDArray[np.float64]:
#     """
#     Apply sensing matrix: equivalent to Phi @ Psi for compressed sensing.
#     Uses efficient JAX-based implementation.
#     """
#     import jax.numpy as jnp

#     if Phi.ndim == 1:
#       Phi = Phi.reshape(1, -1)

#     result = np.zeros((Phi.shape[0], self._dwt_op.shape[0]))

#     for i in range(Phi.shape[0]):
#       phi_row = jnp.array(Phi[i, :])
#       if self._basis:
#         # For basis mode: Phi @ Psi where Psi.times is the inverse transform
#         # We need Phi @ Psi, so we compute the forward transform of Phi
#         coeffs = self._dwt_op.trans(phi_row)
#       else:
#         # For transform mode: apply the forward transform
#         coeffs = self._dwt_op.times(phi_row)

#       result[i, :] = np.array(coeffs)

#     return result.astype(np.float64)

#   def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#     """
#     Transform coefficients back to signal domain.
#     """
#     import jax.numpy as jnp

#     if S.ndim == 1:
#       S = S.reshape(-1, 1)

#     result = np.zeros((S.shape[0], S.shape[1]))

#     for j in range(S.shape[1]):
#       coeffs = jnp.array(S[:, j])
#       if self._basis:
#         # For basis mode: apply the forward operation (times = inverse transform)
#         signal = self._dwt_op.times(coeffs)
#       else:
#         # For transform mode: apply the inverse operation (trans = inverse transform)
#         signal = self._dwt_op.trans(coeffs)

#       result[:, j] = np.array(signal)

#     return result.astype(np.float64)

#   @property
#   def wavelet_name(self) -> str:
#     return self._wavelet

#   @property
#   def decomposition_levels(self) -> int | None:
#     return self._level

#   @property
#   def is_basis_mode(self) -> bool:
#     return self._basis

#   @property
#   def dwt_operator(self):
#     """Access to the underlying cr.sparse DWT operator"""
#     return self._dwt_op


class Gabor(SparsifyingMatrix):
  """
  Gabor basis.
  """

  def __init__(
    self,
    N: int,
    fs: float,
    tf: int = 2,
    ff: int = 4,
    scales: list[int] | None = None,
  ) -> None:
    """
    Create an over-complete *real* Gabor dictionary
    """
    if scales is None:
      scales = [1, 2, 4, 8, 16, 32, 64]

    B = 2.0
    alpha = 0.5 * np.log(0.5 * (B + 1 / B))

    n = np.arange(N)
    atoms: list[npt.NDArray[np.float64]] = []
    for s in scales:
      time_base = s * B * np.sqrt(2 * alpha / np.pi)
      ts = 4 * time_base / tf
      n0_vals = np.arange(0, N, ts)

      freq_base = (np.sqrt(8.0 * np.pi * alpha) / (s * B)) * (fs / 2.0)
      fs_step = freq_base / ff
      # avoid DC / Nyquist
      f0_vals = np.arange(fs_step, fs / 2.0, fs_step)

      for n0 in n0_vals:
        for f0 in f0_vals:
          atom = self._gabor_atom(n, n0, f0, s)
          atoms.append(atom)

    D = np.stack(atoms, axis=1)  # (N × P)
    self._value = D.astype(np.float64)
    self._name = f"Gabor_{len(atoms)}"

  def apply_sensing_matrix(
    self, Phi: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    return Phi @ self.value

  def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return self.value @ S

  def _gabor_atom(
    self, n: list[int], n0: float, f0: float, s: float, phi: float = 0.0
  ) -> npt.NDArray[np.float64]:
    """
    Generate a *real* Gabor atom
    """
    atom = np.exp(-((n - n0) ** 2) / (2.0 * s**2)) * np.sin(
      2.0 * np.pi * f0 * (n - n0) + phi
    )
    atom /= np.linalg.norm(atom)
    return atom.astype(np.float64)
