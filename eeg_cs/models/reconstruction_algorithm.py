from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import spgl1
from sklearn.linear_model import OrthogonalMatchingPursuit as SklearnOMP


class ReconstructionAlgorithm(ABC):
  """
  Abstract class for reconstruction algorithms.
  """

  _name: str

  @property
  def name(self) -> str:
    return self._name

  @abstractmethod
  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]: ...


class Cosamp(ReconstructionAlgorithm):
  def __init__(self, sparsity: int, max_iter: int = 50, tol: float = 1e-6) -> None:
    self.k = int(sparsity)
    self.max_iter = int(max_iter)
    self.tol = float(tol)
    self._name = f"Cosamp_k{self.k}"

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """Recover sparse signal(s) X from measurements Y = Theta @ X.

    Accepts `Y` as shape (m,) or (m, K). Returns X of shape (n,) or (n, K).
    """
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]

    m, n = Theta.shape
    K = Y.shape[1]

    S = np.zeros((n, K), dtype=float)
    for idx in range(K):
      y = Y[:, idx]
      x_hat = self._cosamp_single(y, Theta)
      S[:, idx] = x_hat

    return S

  def _cosamp_single(
    self, y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    m, n = Theta.shape

    # Initialization
    residual = y.copy()
    support: list[int] = []
    x = np.zeros(n, dtype=float)

    for _ in range(self.max_iter):
      # Proxy = Theta^T * residual
      proxy = Theta.T @ residual

      # Identify 2k largest components in magnitude
      omega = np.argpartition(np.abs(proxy), -2 * self.k)[-2 * self.k :]

      # Merge supports
      merged = np.union1d(np.array(support, dtype=int), omega)

      # Restrict Theta to merged support
      if merged.size == 0:
        break

      Theta_merged = Theta[:, merged]

      # Solve least squares on merged support
      try:
        b_merged, *_ = np.linalg.lstsq(Theta_merged, y, rcond=None)
      except np.linalg.LinAlgError:
        b_merged = np.linalg.pinv(Theta_merged) @ y

      # Keep k largest entries from b_merged
      abs_b = np.abs(b_merged)
      if abs_b.size <= self.k:
        # All entries kept
        new_support = merged.astype(int)
      else:
        topk_idx = np.argpartition(abs_b, -self.k)[-self.k :]
        new_support = merged[topk_idx].astype(int)

      # Form new estimate on new_support
      Theta_s = Theta[:, new_support]
      try:
        x_s, *_ = np.linalg.lstsq(Theta_s, y, rcond=None)
      except np.linalg.LinAlgError:
        x_s = np.linalg.pinv(Theta_s) @ y

      # Update full signal estimate
      x.fill(0.0)
      x[new_support] = x_s

      # Update residual
      residual = y - Theta @ x

      # Stopping criteria
      if np.linalg.norm(residual) < self.tol:
        break

      # If support didn't change, we can stop
      if set(new_support.tolist()) == set(support):
        break

      support = new_support.tolist()

    return x


class OrthogonalMatchingPursuit(ReconstructionAlgorithm):
  """
  Orthogonal Matching Pursuit (OMP) reconstruction algorithm using scikit-learn.
  """

  def __init__(
    self, n_nonzero_coefs: int | None = None, tol: float | None = None
  ) -> None:
    self.n_nonzero_coefs = n_nonzero_coefs
    self.tol = tol
    self._name = f"OMP_k{n_nonzero_coefs}"

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    omp = SklearnOMP(n_nonzero_coefs=self.n_nonzero_coefs, tol=self.tol)
    omp.fit(Theta, Y)

    S = np.array(omp.coef_)
    if S.ndim == 1:
      S = S[np.newaxis, :]

    return S.T


# class SimultaneousOrthogonalMatchingPursuit(ReconstructionAlgorithm):
#   def __init__(self, n_nonzero_coefs: int = 10, tol: float = 1e-6) -> None:
#     self.n_nonzero_coefs = n_nonzero_coefs
#     self.tol = tol
#     self._name = f"SOMP_k{n_nonzero_coefs}"

#   def solve(
#     self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
#   ) -> npt.NDArray[np.float64]:
#     """
#     Simultaneous Orthogonal Matching Pursuit (SOMP)
#     """
#     if Y.ndim == 1:
#       # Convert to 2D for consistent processing
#       Y = Y[:, np.newaxis]

#     return self._manual_somp(Y, Theta)

#   def _manual_somp(self, Y: np.ndarray, Theta: np.ndarray) -> np.ndarray:
#     """
#     Manual SOMP implementation as fallback.
#     """
#     _, n = Theta.shape
#     K = Y.shape[1]  # Number of signals

#     # Initialize residual and support
#     R = Y.copy()  # residual: shape (m, K)
#     S: list[int] = []  # support set (list of selected atom indices)

#     # Precompute Theta^T for efficiency
#     ThetaT = Theta.T  # shape (n, m)

#     for _ in range(self.n_nonzero_coefs):
#       # 1) Compute correlations for each atom across all signals
#       #    Use sum of absolute correlations as selection criterion
#       correlations = ThetaT @ R  # shape (n, K)
#       corr_scores = np.sum(np.abs(correlations), axis=1)  # shape (n,)

#       # 2) Mask out already-selected atoms
#       available_mask = np.ones(n, dtype=bool)
#       available_mask[S] = False
#       corr_scores[~available_mask] = -np.inf

#       # 3) Select the atom with maximum correlation score
#       if np.all(corr_scores == -np.inf):
#         break  # No more atoms to select

#       j_best = int(np.argmax(corr_scores))
#       S.append(j_best)

#       # 4) Form the sub-dictionary with selected atoms
#       Theta_S = Theta[:, S]  # shape (m, |S|)

#       # 5) Solve least-squares to get coefficients for all K signals
#       try:
#         X_S = np.linalg.lstsq(Theta_S, Y, rcond=None)[0]  # shape (|S|, K)
#       except np.linalg.LinAlgError:
#         # Fallback to pseudo-inverse if least squares fails
#         X_S = np.linalg.pinv(Theta_S) @ Y

#       # 6) Update residuals: R = Y - Theta_S @ X_S
#       R = Y - Theta_S @ X_S  # shape (m, K)

#       # 7) Check stopping condition
#       residual_norm = np.linalg.norm(R, "fro")  # Frobenius norm for matrix
#       if residual_norm < self.tol:
#         break

#     # 8) Construct the full solution matrix
#     X = np.zeros((n, K))
#     if S:  # Only if we selected any atoms
#       X[S, :] = X_S

#     return X


class SPGL1BasisPursuit(ReconstructionAlgorithm):
  """
  Basis Pursuit using SPGL1 Python library.
  """

  def __init__(self, max_iter: int = 1000, tol: float = 1e-6) -> None:
    self.max_iter = max_iter
    self.tol = tol
    self._name = "BPD"

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]  # Ensure Y is 2D for multiple signals.

    S = []
    for y in Y.T:
      # Solve Basis Pursuit problem
      x, _, _, _ = spgl1.spg_bp(Theta, y, iter_lim=self.max_iter, bp_tol=self.tol)
      S.append(x)

    return np.array(S).T


class SPGL1BasisPursuitDenoising(ReconstructionAlgorithm):
  """
  Basis Pursuit using SPGL1 Python library.
  """

  def __init__(self, max_iter: int = 1000, sigma_factor: float = 0.001) -> None:
    self.sigma_factor = sigma_factor
    self.max_iter = max_iter
    self._name = "BPDN"

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]  # Ensure Y is 2D for multiple signals.

    S = []
    for y in Y.T:
      sigma = self.sigma_factor * np.linalg.norm(y)
      # Solve Basis Pursuit problem
      x, _, _, _ = spgl1.spg_bpdn(Theta, y, sigma=sigma, iter_lim=self.max_iter)
      S.append(x)

    return np.array(S).T


class CVXPBasisPursuit(ReconstructionAlgorithm):
  """
  Basis Pursuit using CVXPY library for convex optimization.
  """

  def __init__(self, solver: str = "CLARABEL", verbose: bool = False) -> None:
    """
    Initialize CVXPY Basis Pursuit solver.

    Parameters
    ----------
    solver : str
        CVXPY solver to use (e.g., 'CLARABEL', 'ECOS', 'OSQP', 'SCS').
    verbose : bool
        Whether to print solver output.
    """
    self.solver = solver
    self.verbose = verbose
    self._name = f"CVXP_BP_{solver}"

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Solve Basis Pursuit problem: minimize ||x||_1 subject to Theta @ x = y

    Parameters
    ----------
    Y : ndarray
        Measurement vector(s) of shape (m,) or (m, K) for K signals.
    Theta : ndarray
        Sensing matrix of shape (m, n).

    Returns
    -------
    ndarray
        Reconstructed sparse signal(s) of shape (n,) or (n, K).
    """
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]  # Ensure Y is 2D for multiple signals.

    m, n = Theta.shape
    K = Y.shape[1]

    X = cp.Variable((n, K))

    objective = cp.Minimize(cp.sum(cp.norm(X, 1, axis=0)))
    constraints = [Theta @ X == Y]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=self.solver, verbose=self.verbose)

    return X.value


class CVXPBasisPursuitIndividual(ReconstructionAlgorithm):
  """
  Basis Pursuit using CVXPY library for convex optimization.
  This version solves the optimization problem for each signal individually.
  """

  def __init__(
    self, solver: str = "CLARABEL", max_iter: int = 100000, verbose: bool = False
  ) -> None:
    """
    Initialize CVXPY Basis Pursuit solver for individual signal processing.

    Parameters
    ----------
    solver : str
        CVXPY solver to use (e.g., 'CLARABEL', 'ECOS', 'OSQP', 'SCS').
    verbose : bool
        Whether to print solver output.
    """
    self.solver = solver
    self.max_iter = max_iter
    self.verbose = verbose
    self._name = f"CVXP_BP_IND_{solver}"

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Solve Basis Pursuit problem for each signal individually: minimize ||x||_1 subject to Theta @ x = y

    Parameters
    ----------
    Y : ndarray
        Measurement vector(s) of shape (m,) or (m, K) for K signals.
    Theta : ndarray
        Sensing matrix of shape (m, n).

    Returns
    -------
    ndarray
        Reconstructed sparse signal(s) of shape (n,) or (n, K).
    """
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]  # Ensure Y is 2D for multiple signals.

    m, n = Theta.shape
    K = Y.shape[1]

    # Initialize result array
    X_result = np.zeros((n, K))

    # Solve for each signal individually
    for k in range(K):
      y_k = Y[:, k]

      # Define optimization variable for single signal
      x = cp.Variable(n)

      # Define objective: minimize L1 norm
      objective = cp.Minimize(cp.norm(x, 1))

      # Define constraints: Theta @ x = y_k
      constraints = [Theta @ x == y_k]

      # Create and solve problem
      problem = cp.Problem(objective, constraints)
      problem.solve(solver=self.solver, verbose=self.verbose, max_iter=self.max_iter)

      X_result[:, k] = x.value

    return X_result


class CVXPBasisPursuitDenoisingIndividual(ReconstructionAlgorithm):
  """
  Basis Pursuit Denoising using CVXPY library for convex optimization.
  This version solves the optimization problem for each signal individually.
  """

  def __init__(
    self,
    solver: str = "CLARABEL",
    sigma_factor: float = 0.001,
    verbose: bool = False,
  ) -> None:
    """
    Initialize CVXPY Basis Pursuit Denoising solver for individual signal processing.

    Parameters
    ----------
    solver : str
        CVXPY solver to use (e.g., 'CLARABEL', 'ECOS', 'OSQP', 'SCS').
    verbose : bool
        Whether to print solver output.
    sigma_factor : float
        Factor to compute noise level sigma = sigma_factor * ||y||_2.
    """
    self.solver = solver
    self.verbose = verbose
    self.sigma_factor = sigma_factor
    self._name = f"CVXP_BPDN_IND_{solver}"

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Solve Basis Pursuit Denoising problem for each signal individually:
    minimize ||x||_1 subject to ||Theta @ x - y||_2 <= sigma

    Parameters
    ----------
    Y : ndarray
        Measurement vector(s) of shape (m,) or (m, K) for K signals.
    Theta : ndarray
        Sensing matrix of shape (m, n).

    Returns
    -------
    ndarray
        Reconstructed sparse signal(s) of shape (n,) or (n, K).
    """
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]  # Ensure Y is 2D for multiple signals.

    m, n = Theta.shape
    K = Y.shape[1]

    # Initialize result array
    X_result = np.zeros((n, K))

    # Solve for each signal individually
    for k in range(K):
      y_k = Y[:, k]

      # Compute noise level for this signal
      sigma = self.sigma_factor * np.linalg.norm(y_k)

      # Define optimization variable for single signal
      x = cp.Variable(n)

      # Define objective: minimize L1 norm
      objective = cp.Minimize(cp.norm(x, 1))

      # Define constraints: ||Theta @ x - y_k||_2 <= sigma
      constraints = [cp.norm(Theta @ x - y_k, 2) <= sigma]

      # Create and solve problem
      problem = cp.Problem(objective, constraints)
      problem.solve(solver=self.solver, verbose=self.verbose)

      X_result[:, k] = x.value

    return X_result
