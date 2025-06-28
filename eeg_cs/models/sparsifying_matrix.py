from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pywt
from scipy.fftpack import dct, dst, idct, idst


class SparsifyingMatrix(ABC):
    """
    Abstract class for sparse basis.
    """

    _value: npt.NDArray[np.float64]

    @property
    def value(self) -> npt.NDArray[np.float64]:
        return self._value

    @abstractmethod
    def apply_sensing_matrix(
        self, Phi: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

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

    def __init__(self, N: int):
        """
        Create a DCT basis of size N.
        """
        # DCT basis matrix for coherence calculation
        self._value = idct(np.eye(N), norm="ortho", axis=0)

    def apply_sensing_matrix(self, Phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Equivalent to Phi @ self.value
        return dct(Phi, norm="ortho")

    def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return idct(S, norm="ortho", axis=0)


class DST(SparsifyingMatrix):
    """
    Discrete Sine Transform (DST) basis.
    """

    def __init__(self, N: int) -> None:
        """
        Create a DST basis of size N.
        """
        # DST basis matrix for coherence calculation
        self._value = idst(np.eye(N), type=2, norm="ortho", axis=0)

    def apply_sensing_matrix(self, Phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Equivalent to Phi @ self.value
        return dst(Phi, type=2, norm="ortho")

    def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return idst(S, type=2, norm="ortho", axis=0)


class CS(SparsifyingMatrix):
    """
    Overcomplete basis combining DCT (C) and DST (S).
    """

    def __init__(self, N: int):
        identity = np.eye(N)
        dct_basis = idct(identity, norm="ortho", axis=0)
        dst_basis = idst(identity, type=2, norm="ortho", axis=0)

        self._value = np.hstack((dct_basis, dst_basis))

    def apply_sensing_matrix(self, Phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return Phi @ self.value

    def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.value @ S


class ICS(SparsifyingMatrix):
    """
    Overcomplete basis combining Impulse (I), DCT (C), and DST (S).
    """

    def __init__(self, N: int):
        identity = np.eye(N)
        dct_basis = idct(identity, norm="ortho", axis=0)
        dst_basis = idst(identity, type=2, norm="ortho", axis=0)

        self._value = np.hstack((identity, dct_basis, dst_basis))

    def apply_sensing_matrix(self, Phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return Phi @ self.value

    def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.value @ S


# class KSVD(SparsifyingMatrix):
#     """
#     Overcomplete learned dictionary.
#     """
#     _id: int = time.time_ns()

#     def __init__(self, training_data: npt.NDArray[np.float64], n_atoms: int = 100, n_coefficients: int = 30, max_iter: int = 80):
#     #     from ksvd import ksvd
#     #     dictionary,_,_ = ksvd(training_data, n_atoms, n_coefficients, initial_D=None,
#     # maxiter=max_iter, etol=1e-10, approx=False, debug=True)
#     #     self._value = dictionary

#         # dl = MiniBatchDictionaryLearning(
#         #     n_components=n_atoms,
#         #     transform_n_nonzero_coefs=n_coefficients,
#         #     fit_algorithm='lars',      # MOD-style update; very similar to K-SVD in practice
#         #     transform_algorithm='omp', # Orthogonal Matching Pursuit for sparse coding
#         #     max_iter=max_iter,
#         #     random_state=random_state,
#         #     n_jobs=-1,  # Use all available CPU cores
#         # )
#         # self._value = dl.fit(training_data).components_.T

#         KSVD_model = KSVD(K=n_atoms, T0=n_coefficients, max_iter=max_iter)
#         KSVD_model.fit_with_mean(training_data, verbose=True)
#         self._value = KSVD_model.D

#     def apply_sensing_matrix(self, Phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#         return Phi @ self.value

#     def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#         return self.value @ S

#     @classmethod
#     def load(cls, file_path: str):
#         """
#         Load data from a file.
#         This method should be implemented by subclasses.
#         """
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File {file_path} does not exist.")

#         with open(file_path, "rb") as f:
#             obj = pickle.load(f)
#             if not isinstance(obj, cls):
#                 raise TypeError(f"Expected object of type {cls.__name__}, got {type(obj).__name__}.")

#             return obj

#     def save(self, folder_path: str):
#         n_atoms = self.value.shape[1]
#         file_path = os.path.join(folder_path, f"{self.id}_ksvd_{n_atoms}.pkl")
#         with open(file_path, "wb") as f:
#             pickle.dump(self, f)


#     @property
#     def id(self) -> int:
#         return self._id


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
    ):
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
            fs_step = freq_base / ff  # Eq. 10 → frequency step (Hz)
            f0_vals = np.arange(fs_step, fs / 2.0, fs_step)  # avoid DC / Nyquist

            for n0 in n0_vals:
                for f0 in f0_vals:
                    atom = self._gabor_atom(n, n0, f0, s)
                    atoms.append(atom)

        D = np.stack(atoms, axis=1)  # (N × P)
        self._value = D.astype(np.float64)

    def apply_sensing_matrix(self, Phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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


class Wavelet(SparsifyingMatrix):
    """
    Overcomplete wavelet dictionary built by inverting each coefficient index
    in a standard DWT decomposition.

    - N: signal length
    - wavelet: any PyWavelets wavelet name (e.g. 'db4', 'sym6', 'meyr', etc.)
    - level: number of decomposition levels (defaults to max possible)
    """

    def __init__(self, N: int, wavelet: str = "db4", level: int | None = None):
        self.wavelet = wavelet
        max_lev = pywt.dwt_max_level(N, wavelet)
        self.level = max_lev if level is None else min(level, max_lev)

        # get coefficient‐to‐array mapping for sizing
        zero_coeffs = pywt.wavedec(np.zeros(N), wavelet, level=self.level)
        arr, self._slices = pywt.coeffs_to_array(zero_coeffs)
        K = arr.size  # total number of wavelet coefficients

        # build dictionary: each column is the reconstruction of a unit coefficient
        D = np.zeros((N, K), dtype=float)
        for k in range(K):
            unit = np.zeros(K, dtype=float)
            unit[k] = 1.0
            coeffs_k = pywt.array_to_coeffs(unit, self._slices, output_format="wavedec")
            rec = pywt.waverec(coeffs_k, wavelet)
            D[:, k] = rec[:N]  # truncate any padding

        self._value = D

    def apply_sensing_matrix(self, Phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return Phi @ self.value

    def transform(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.value @ S
