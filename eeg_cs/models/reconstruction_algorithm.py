from abc import ABC, abstractmethod

import cr.sparse.cvx.spgl1 as crspgl1
import cvxpy as cp
import numpy as np
import numpy.typing as npt
from cr.sparse import lop
from sklearn.linear_model import OrthogonalMatchingPursuit as SklearnOMP


class ReconstructionAlgorithm(ABC):
  """
  Abstract class for reconstruction algorithms.
  """

  @abstractmethod
  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]: ...


class BasisPursuit(ReconstructionAlgorithm):
  """
  Basis Pursuit.
  """

  def __init__(self, **kwargs: dict[str, bool | int | str]):
    """
    Parameters
    ----------
    kwargs : dict
        Additional arguments for the optimization problem.
    """
    self.kwargs = kwargs or {}

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]

    S = []
    for y in Y.T:
      s = self._solve_single(y, Theta)
      S.append(s)

    return np.array(S).T

  def _solve_single(
    self, y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    alpha = cp.Variable(Theta.shape[1])
    objective = cp.Minimize(cp.norm1(alpha))
    constraints = [y == Theta @ alpha]
    prob = cp.Problem(objective, constraints)
    prob.solve(**self.kwargs)
    if prob.status != cp.OPTIMAL:
      raise ValueError("The optimization problem is not solvable.")

    return np.asarray(alpha.value, dtype=np.float32)


class BasisPursuitDenoising(ReconstructionAlgorithm):
  """
  Basis Pursuit Denoising (BPDN).
  Solves: min ||alpha||_1  s.t.  ||Y - Theta @ alpha||_2 <= epsilon
  """

  def __init__(self, epsilon: float = 1e-3, **kwargs):
    """
    Parameters
    ----------
    epsilon : float
        Allowed reconstruction error (noise level).
    kwargs : dict
        Additional arguments for the optimization solver.
    """
    self.epsilon = epsilon
    self.kwargs = kwargs or {}

  def solve(self, Y: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]

    S = []
    for y in Y.T:
      s = self._solve_single(y, Theta)
      S.append(s)

    return np.array(S).T

  def _solve_single(self, y: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    alpha = cp.Variable(Theta.shape[1])
    objective = cp.Minimize(cp.norm1(alpha))
    constraints = [cp.norm2(y - Theta @ alpha) <= self.epsilon]
    prob = cp.Problem(objective, constraints)
    prob.solve(**self.kwargs)
    if prob.status != cp.OPTIMAL:
      raise ValueError("The optimization problem is not solvable.")

    return np.asarray(alpha.value, dtype=np.float32)


class OrthogonalMatchingPursuit(ReconstructionAlgorithm):
  """
  Orthogonal Matching Pursuit (OMP) reconstruction algorithm using scikit-learn.
  """

  def __init__(self, n_nonzero_coefs: int = 10, tol: float = 1e-6):
    self.n_nonzero_coefs = n_nonzero_coefs
    self.tol = tol

  def solve(self, Y: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """
    Solve the sparse signal(s) using OMP.

    Parameters
    ----------
    Y : np.ndarray
        The observed measurements (m,) or (m, n_signals).
    Theta : np.ndarray
        The sensing matrix (m, n).

    Returns
    -------
    np.ndarray
        The reconstructed sparse signal(s) (n, n_signals).
    """
    if Y.ndim == 1:
      Y = Y[:, np.newaxis]

    S = []
    for y in Y.T:
      omp = SklearnOMP(n_nonzero_coefs=self.n_nonzero_coefs)
      omp.fit(Theta, y)
      S.append(omp.coef_)

    return np.array(S).T


# class OrthogonalMatchingPursuit(ReconstructionAlgorithm):
#     """
#     Orthogonal Matching Pursuit (OMP) reconstruction algorithm.
#     """

#     def __init__(self, max_iter: int = 100, tol: float = 1e-6):
#         """
#         Parameters
#         ----------
#         max_iter : int
#             Maximum number of iterations for the OMP algorithm.
#         tol : float
#             Tolerance for the residual norm to stop the algorithm.
#         """
#         self.max_iter = max_iter
#         self.tol = tol

#     def solve(self, Y: np.ndarray, Theta: np.ndarray) -> np.ndarray:
#         """
#         solve the sparse signal using OMP.

#         Parameters
#         ----------
#         y : np.ndarray
#             The observed measurements (m-dimensional vector).
#         Phi : np.ndarray
#             The sensing matrix (m x n matrix).

#         Returns
#         -------
#         np.ndarray
#             The reconstructed sparse signal (n-dimensional vector).
#         """
#         if Y.ndim == 1:
#             Y = Y[:, np.newaxis]

#         S = []
#         for y in Y.T:
#             s = self._solve_single(y, Theta)
#             S.append(s)

#         return np.array(S).T

#     def _solve_single(self, y: np.ndarray, Theta: np.ndarray) -> np.ndarray:
#         """
#         Solve the sparse signal using OMP for a single measurement.

#         Parameters
#         ----------
#         y : np.ndarray
#             The observed measurements (m-dimensional vector).
#         Phi : np.ndarray
#             The sensing matrix (m x n matrix).

#         Returns
#         -------
#         np.ndarray
#             The reconstructed sparse signal (n-dimensional vector).
#         """
#         m, n = Theta.shape
#         ThetaT = Theta.T

#         # Initialize residual and support
#         residual = y.copy()
#         support = []

#         for _ in range(self.max_iter):
#             # Step 1: Find the column of Phi most correlated with the residual
#             correlations = ThetaT @ residual
#             best_index = np.argmax(np.abs(correlations))
#             support.append(best_index)

#             # Step 2: Solve least squares problem to update the solution
#             Theta_support = Theta[:, support]
#             s_support = np.linalg.lstsq(Theta_support, y, rcond=None)[0]

#             # Step 3: Update the residual
#             residual = y - Theta_support @ s_support

#             # Check stopping condition
#             if np.linalg.norm(residual) < self.tol:
#                 break

#         # Step 4: Construct the full solution vector
#         s = np.zeros(n)
#         s[support] = s_support

#         return s


class SimultaneousOrthogonalMatchingPursuit(ReconstructionAlgorithm):
  """
  Simultaneous Orthogonal Matching Pursuit (SOMP) reconstruction algorithm.
  """

  def __init__(self, max_iter: int = 100, tol: float = 1e-6):
    self.max_iter = max_iter
    self.tol = tol

  def solve(self, Y: np.ndarray, Theta: np.ndarray):
    """
    Simultaneous Orthogonal Matching Pursuit (SOMP)

    Parameters
    ----------
    Y   : array, shape (m, K)
        Matrix of K measurement vectors (each column is one measurement).
    Phi : array, shape (m, n)
        Dictionary matrix whose columns are the atoms.
    s   : int
        Sparsity level: number of atoms to select.

    Returns
    -------
    S   : list of int
        Indices of the selected atoms (size s).
    X   : array, shape (s, K)
        Coefficient matrix (only the rows corresponding to S are nonzero).
    """
    # Initialize residual and support
    R = Y.copy()  # residual: shape (m, K)
    S: list[int] = []  # support set (list of selected indices)

    # Precompute Phi^T for efficiency
    ThetaT = Theta.T  # shape (n, m)

    for _ in range(self.max_iter):
      # 1) Compute (absolute) correlations for each atom j: sum over all K signals
      #    corr[j] = || (Theta[:, j].T @ R) ||_1
      # Using matrix form: (Theta^T @ R) is (n x K), then take abs and sum across columns.
      corr = np.sum(np.abs(ThetaT @ R), axis=1)  # shape (n,)

      # 2) Mask out already-selected atoms (set their score to -inf)
      mask = np.ones_like(corr, dtype=bool)
      mask[S] = False
      corr_masked = np.where(mask, corr, -np.inf)

      # 3) Pick the atom with maximum correlation
      j_best = int(np.argmax(corr_masked))
      S.append(j_best)

      # 4) Form the sub-dictionary with selected atoms
      Phi_S = Theta[:, S]  # shape (m, |S|)

      # 5) Solve least-squares to get coefficients for all K signals:
      #    X_S = Phi_S^+ Y, where Phi_S^+ is the Moore–Penrose pseudoinverse
      X_S = np.linalg.pinv(Phi_S) @ Y  # shape (|S|, K)

      # 6) Update residuals: R = Y - Phi_S X_S
      R = Y - Phi_S @ X_S  # shape (m, K)

      # 7) Check stopping condition
      if np.linalg.norm(R) < self.tol:
        break

    # 8) Construct the full solution vector
    # X: full n×K coefficient matrix, but only rows in S are nonzero.
    X = np.zeros((Theta.shape[1], Y.shape[1]))
    X[S, :] = X_S

    return X


class SPGL1BasisPursuit(ReconstructionAlgorithm):
  """
  Basis Pursuit using SPGL1 from cr-sparse.
  """

  def __init__(self, max_iter: int = 1000):
    self.max_iter = max_iter

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    Theta_op = lop.matrix(Theta)
    options = crspgl1.SPGL1Options(max_iters=self.max_iter)

    if Y.ndim == 1:
      Y = Y[:, np.newaxis]  # type: ignore

    S = []
    for y in Y.T:
      sol = crspgl1.solve_bp(Theta_op, y, options=options)
      s = sol.x
      S.append(s)

    return np.array(S).T


class SPGL1BasisPursuitDenoising(ReconstructionAlgorithm):
  """
  Basis Pursuit Denoising using SPGL1 from cr-sparse.
  """

  def __init__(self, sigma_factor: float = 0.01, max_iter: int = 1000):
    """
    Parameters
    ----------
    sigma_factor : float
        Fraction of measurement norm to use as noise bound (sigma).
    max_iter : int
        Maximum number of SPGL1 iterations.
    """
    self.sigma_factor = sigma_factor
    self.max_iter = max_iter

  def solve(
    self, Y: npt.NDArray[np.float64], Theta: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    """
    Solves min ||alpha||_1 s.t. ||Y - Theta @ alpha||_2 <= sigma using SPGL1.

    Parameters
    ----------
    Y : npt.NDArray[np.float64]
        Measurements (shape: (m,) or (m, n_signals))
    Theta : operator or ndarray
        Sensing matrix or operator (should support .times and .trans)
    x0 : npt.NDArray[np.float64], optional
        Initial guess for the solution.

    Returns
    -------
    alpha : npt.NDArray[np.float64]
        Solution coefficients (shape: (n,) or (n, n_signals))
    """
    # If Theta is a numpy array, wrap it as a cr-sparse operator
    Theta_op = lop.matrix(Theta)
    options = crspgl1.SPGL1Options(max_iters=self.max_iter)

    if Y.ndim == 1:
      Y = Y[:, np.newaxis]  # type: ignore

    S = []
    for y in Y.T:
      sigma = self.sigma_factor * np.linalg.norm(y)
      sol = crspgl1.solve_bpic_jit(Theta_op, y, sigma, options)
      S.append(sol.x)

    return np.array(S).T
