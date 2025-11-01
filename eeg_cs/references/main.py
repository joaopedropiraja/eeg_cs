import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# -------- Generate Sparse Signal in DCT Domain --------
n = 256
k = 2  # sparsity
np.random.seed(42)
alpha_true = np.zeros(n)
alpha_true[np.random.choice(n, k, replace=False)] = np.random.randn(k)
x = idct(alpha_true, norm="ortho")

# -------- Measurement Matrices --------
m = 150
A_rand = np.random.randn(m, n)  # random projection
indices = np.sort(np.random.choice(n, m, replace=False))
A_dirac = np.eye(n)[indices, :]  # under-sampling (subsampled identity)

# -------- Measurements --------
y_cs = A_rand @ x
y_under = A_dirac @ x

# -------- DCT Basis --------
Psi = np.eye(n)
Psi_dct = dct(Psi, norm="ortho")


# -------- 1. Lasso Recovery --------
def recover_lasso(y, A, Psi, lam=0.01):
  A_eff = A @ Psi.T
  lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
  lasso.fit(A_eff, y)
  return lasso.coef_


alpha_cs_lasso = recover_lasso(y_cs, A_rand, Psi_dct)
alpha_under_lasso = recover_lasso(y_under, A_dirac, Psi_dct)
x_cs_lasso = idct(alpha_cs_lasso, norm="ortho")
x_under_lasso = idct(alpha_under_lasso, norm="ortho")


# -------- 2. CoSaMP Recovery --------
def cosamp(Phi, u, s, tol=1e-4, max_iter=100):
  m, n = Phi.shape
  a = np.zeros(n)
  v = u.copy()
  T = set()
  for _ in range(max_iter):
    y = Phi.T @ v
    Omega = set(np.abs(y).argsort()[-2 * s :])
    T = T.union(Omega)
    Phi_T = Phi[:, list(T)]
    b, _, _, _ = np.linalg.lstsq(Phi_T, u, rcond=None)
    b_hat = np.zeros(n)
    b_sorted_indices = np.argsort(np.abs(b))[-s:]
    for idx in b_sorted_indices:
      b_hat[list(T)[idx]] = b[idx]
    a = b_hat
    v = u - Phi @ a
    if np.linalg.norm(v) < tol:
      break
  return a


alpha_cs_cosamp = cosamp(A_rand @ Psi_dct.T, y_cs, s=k)
alpha_under_cosamp = cosamp(A_dirac @ Psi_dct.T, y_under, s=k)
x_cs_cosamp = idct(alpha_cs_cosamp, norm="ortho")
x_under_cosamp = idct(alpha_under_cosamp, norm="ortho")


# -------- 3. Basis Pursuit Recovery --------
def basis_pursuit(Phi, y):
  n = Phi.shape[1]
  alpha = cp.Variable(n)
  objective = cp.Minimize(cp.norm1(alpha))
  constraints = [Phi @ alpha == y]
  prob = cp.Problem(objective, constraints)
  prob.solve()
  return alpha.value


alpha_cs_bp = basis_pursuit(A_rand @ Psi_dct.T, y_cs)
alpha_under_bp = basis_pursuit(A_dirac @ Psi_dct.T, y_under)
x_cs_bp = idct(alpha_cs_bp, norm="ortho")
x_under_bp = idct(alpha_under_bp, norm="ortho")


# -------- RMSE Evaluation --------
def print_rmse(label, x_true, x_rec_cs, x_rec_under):
  print(f"\n{label}:")
  print(f"  RMSE (CS)         : {np.sqrt(mean_squared_error(x_true, x_rec_cs)):.4f}")
  print(f"  RMSE (Under-samp) : {np.sqrt(mean_squared_error(x_true, x_rec_under)):.4f}")


print_rmse("LASSO", x, x_cs_lasso, x_under_lasso)
print_rmse("CoSaMP", x, x_cs_cosamp, x_under_cosamp)
print_rmse("Basis Pursuit", x, x_cs_bp, x_under_bp)


def plot_results(x, x_cs, x_under, alpha_true, alpha_cs, alpha_under, title):
  plt.figure(figsize=(14, 6))

  # Frequency domain
  plt.subplot(1, 2, 1)
  plt.stem(alpha_true, linefmt="k-", markerfmt="ko", basefmt="k-", label="Original")
  plt.stem(alpha_cs, linefmt="b--", markerfmt="bo", basefmt=" ", label="CS")
  plt.stem(
    alpha_under, linefmt="r:", markerfmt="ro", basefmt=" ", label="Under-sampled"
  )
  plt.title(f"{title} - DCT Coefficients")
  plt.xlabel("Index")
  plt.ylabel("Coefficient Value")
  plt.legend()
  plt.grid()

  # Time domain
  plt.subplot(1, 2, 2)
  plt.plot(x, "k-", label="Original")
  plt.plot(x_cs, "b--", label="CS")
  plt.plot(x_under, "r:", label="Under-sampled")
  plt.title(f"{title} - Time Domain Signal")
  plt.xlabel("Time Index")
  plt.ylabel("Amplitude")
  plt.legend()
  plt.grid()

  plt.tight_layout()
  plt.show()


# Plot for Lasso
plot_results(
  x, x_cs_lasso, x_under_lasso, alpha_true, alpha_cs_lasso, alpha_under_lasso, "Lasso"
)

# Plot for CoSaMP
plot_results(
  x,
  x_cs_cosamp,
  x_under_cosamp,
  alpha_true,
  alpha_cs_cosamp,
  alpha_under_cosamp,
  "CoSaMP",
)

# Plot for Basis Pursuit
plot_results(
  x, x_cs_bp, x_under_bp, alpha_true, alpha_cs_bp, alpha_under_bp, "Basis Pursuit"
)
