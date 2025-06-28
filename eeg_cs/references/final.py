# Bibliotecas
import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct
from sklearn.linear_model import OrthogonalMatchingPursuit
import cvxpy as cp

# Arquivo de exemplo da base "CHB-MIT Scalp EEG Database" 
raw = mne.io.read_raw_edf('./files/chb01_14.edf')

# print(raw)
print(raw.info)

# raw.plot(duration=4)

# Captura de uma janela de 1s de um canal específico
fs = raw.info['sfreq']

start_time = 9
end_time = 10

ch_name = raw.ch_names[1]

x = raw.get_data(tmin=start_time, tmax=end_time, picks=(ch_name)).flatten()
n = x.size
t = np.linspace(start_time, end_time, num=n)

# n = 256 # points in high resolution signal
# t = np.linspace(0,1,n)
# f1 = 115
# f2 = 40
# f3 = 45
# x = np.cos(2 * f1 * np.pi * t) + np.cos(2 * f2 * np.pi * t) +  np.cos(2 * f3 * np.pi * t)

X = 2 * np.abs(dct(x, norm='ortho')) / n
freq = np.fft.fftfreq(n, 1 / fs)


# Plota a janela de 1 segundo (256 amostras)
# fig, ax = plt.subplots(2, figsize=[15, 5])
# ax[0].plot(x, color='black')
# ax[0].set_xlabel('Tempo (s)')
# ax[0].set_ylabel('Amplitude')
# ax[0].set_title('Sinal EEG')
# ax[1].stem(X, markerfmt='.', basefmt='None')
# ax[1].set_xlabel('Frequência (Hz)')
# ax[1].set_ylabel('Amplitude')
# ax[1].set_title('Transformada Discreta do Cosseno (DCT)')

# plt.grid()
# plt.show()

# Aquisição dos dados (sensing matrix)
cr = 2
m = int(n / cr)

# -----------Gaussian random projection-----------
# rng = np.random.default_rng(42)
# Theta_gauss = rng.standard_normal((m, n)) / np.sqrt(m)
# Theta_gauss = np.random.randn(m, n)  # random projection
# y = Theta_gauss @ x
# Phi = dct(Theta_gauss, norm='ortho')

# -----------Subsampled identity-----------
# indices = np.sort(np.random.choice(n, m, replace=False))
# Theta_dirac = np.eye(n)[indices, :]  # under-sampling (subsampled identity)
# y = Theta_dirac @ x
# Phi = dct(Theta_dirac, norm='ortho')

# -----------Binary permuted block diagonal matrix-----------S
I = np.eye(m, dtype=int)
Theta_BPBD = np.repeat(I, repeats=cr, axis=1)
y = Theta_BPBD @ x
Phi = dct(Theta_BPBD, norm='ortho')

# alpha = cp.Variable(n)
# objective = cp.Minimize(cp.norm1(alpha))
# constraints = [Phi @ alpha == y]
# prob = cp.Problem(objective, constraints)
# prob.solve()
# s = alpha.value

# omp = OrthogonalMatchingPursuit(n_nonzero_coefs=m)
# omp.fit(Phi, y)
# s = omp.coef_

def omp(T: np.ndarray,
        y: np.ndarray,
        sparsity: int,
        tol: float = 1e-6) -> np.ndarray:
    """
    Orthogonal Matching Pursuit (OMP).

    Args:
        T (m×n): Sensing matrix (often Phi·Psi).
        y (m,):   Measurement vector.
        sparsity: Maximum number of nonzero coefficients (max iterations).
        tol:      Residual‐norm threshold for early stopping.

    Returns:
        alpha (n,): Recovered sparse coefficient vector.
    """
    m, n = T.shape
    # 1) Initialize
    residual = y.copy()               # r^0 = y
    support = []                      # Ω = ∅
    alpha = np.zeros(n)               # sparse vector S
    # 2) Iterate
    for i in range(sparsity):
        # 2a) Correlate residual with all atoms
        proj = T.T @ residual
        # 2b) Select index of maximum absolute projection
        idx = np.argmax(np.abs(proj))
        support.append(idx)
        # 2c) Form submatrix with selected atoms
        T_sub = T[:, support]
        # 2d) Solve least‐squares on support
        beta_sub, *_ = np.linalg.lstsq(T_sub, y, rcond=None)
        # 2e) Update residual
        residual = y - T_sub @ beta_sub
        # 2f) Early stop criterion
        if np.linalg.norm(residual) < tol:
            break
    # 3) Fill in the recovered coefficients
    alpha[support] = beta_sub
    return alpha

s = omp(Phi, y, sparsity=m)


# def cosamp(Phi, u, s, tol=1e-4, max_iter=100):
#     m, n = Phi.shape
#     a = np.zeros(n)
#     v = u.copy()
#     T = set()
#     for _ in range(max_iter):
#         y = Phi.T @ v
#         Omega = set(np.abs(y).argsort()[-2*s:])
#         T = T.union(Omega)
#         Phi_T = Phi[:, list(T)]
#         b, _, _, _ = np.linalg.lstsq(Phi_T, u, rcond=None)
#         b_hat = np.zeros(n)
#         b_sorted_indices = np.argsort(np.abs(b))[-s:]
#         for idx in b_sorted_indices:
#             b_hat[list(T)[idx]] = b[idx]
#         a = b_hat
#         v = u - Phi @ a
#         if np.linalg.norm(v) < tol:
#             break
#     return a

# s = cosamp(Phi, y, s=128)

x_hat = idct(s, norm='ortho')
X_hat = 2 * np.abs(dct(x_hat, norm='ortho')) / n

# Plotando os resultados
# fig, ax = plt.subplots(2, 2, figsize=[15, 5])

# ax[0, 0].plot(t, x, color='black')
# ax[0, 0].set_xlabel('Tempo (s)')
# ax[0, 0].set_ylabel('Amplitude')
# ax[0, 0].set_title('Sinal original')
# ax[0, 0].grid()

# ax[0, 1].stem(X, basefmt='None')
# ax[0, 1].set_xlabel('Frequência (Hz)')
# ax[0, 1].set_ylabel('Amplitude')
# ax[0, 1].set_title('DFT do sinal original')
# ax[0, 1].grid()

# ax[1, 0].plot(t, x_hat, color='red')
# ax[1, 0].set_xlabel('Tempo (s)')
# ax[1, 0].set_ylabel('Amplitude')
# ax[1, 0].set_title('Sinal reconstruído')
# ax[1, 0].grid()

# ax[1, 1].stem(X_hat, linefmt='r', markerfmt='r', basefmt='None')
# ax[1, 1].set_xlabel('Frequência (Hz)')
# ax[1, 1].set_ylabel('Amplitude')
# ax[1, 1].set_title('DFT do sinal reconstruído')
# ax[1, 1].grid()

# plt.show()

fig, ax = plt.subplots(figsize=[15, 5])
ax.plot(t, x, 'k', label='Sinal original')
ax.plot(t, x_hat, 'r', linestyle='dotted', label='Sinal reconstruído')
ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Sinal original x Sinal reconstruído')
ax.legend(loc='upper right', framealpha=1)
ax.grid()

plt.show()

# # ————— Figure 3: overlay time-series —————
# import plotly.graph_objects as go


# fig3 = go.Figure()
# fig3.add_trace(
#     go.Scatter(x=t, y=x, mode='lines', name='Sinal original', line=dict(color='blue', dash='solid'))
# )
# fig3.add_trace(
#     go.Scatter(x=t, y=x_hat, mode='lines', name='Sinal reconstruído',
#             line=dict(color='red', dash='dash'))
# )

# fig3.update_layout(
#     title='Sinal original x Sinal reconstruído',
#     xaxis_title='Tempo (s)',
#     yaxis_title='Amplitude',
#     legend=dict(x=0.75, y=1),
#     width=1000, height=400
# )

# fig3.show()    

def sndr(x: np.ndarray, x_hat: np.ndarray) -> float:
    """
    Returns:
        float: Signal to Noise and Distortion Ratio SNDR (dB).
    """
    num = np.linalg.norm(x)**2
    den = np.linalg.norm(x - x_hat)**2
    if den == 0:
        return np.inf
    return 10 * np.log10(num / den)

def nmse(x: np.ndarray, x_hat: np.ndarray) -> float:
    num = np.linalg.norm(x - x_hat)**2
    den = np.linalg.norm(x)**2
    if (den == 0):
        return np.inf
    return float(num / den)

print(f"SNDR: {sndr(x, x_hat):.10f} dB")
print(f"NMSE: {nmse(x, x_hat):.10f}")