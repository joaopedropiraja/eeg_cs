import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from eeg_cs.models.compressed_sensing import CompressedSensing
from eeg_cs.models.loader import CHBMITLoader
from eeg_cs.models.reconstruction_algorithm import (
  CVXPBasisPursuitIndividual,
  ReconstructionAlgorithm,
  SPGL1BasisPursuit,
)
from eeg_cs.models.sensing_matrix import (
  BinaryPermutedBlockDiagonal,
  Gaussian,
  SensingMatrix,
  Undersampled,
)
from eeg_cs.models.sparsifying_matrix import DCT, Gabor, SparsifyingMatrix, Wavelet

type CS_Architecture = tuple[
  SensingMatrix, SparsifyingMatrix, ReconstructionAlgorithm, float
]


def plot_reconstructions(
  x: npt.NDArray[np.float64],
  cs_architectures: list[CS_Architecture],
  start_time_idx: int,
  downsampled_fs: float,
) -> None:
  segment_length = len(x)

  start_time_s = start_time_idx / downsampled_fs
  time_axis = start_time_s + np.arange(segment_length) / downsampled_fs

  n_architectures = len(cs_architectures)
  if n_architectures % 2 == 1:
    ncols = 2
    nrows = (n_architectures + 1) // 2
  else:
    ncols = 1
    nrows = n_architectures + 1

  _, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(12 * ncols, 2 * nrows),
    sharex=True,
    sharey=True,
  )

  if nrows == 1 and ncols == 1:
    axes = np.array([[axes]])
  elif nrows == 1:
    axes = axes.reshape(1, -1)
  elif ncols == 1:
    axes = axes.reshape(-1, 1)

  if ncols == 2:
    ax_orig = plt.subplot(nrows, ncols, 1)
    ax_orig.plot(time_axis, x, color="b", label="Sinal Original", linewidth=1.5)
    ax_orig.set_ylabel("Amplitude")
    ax_orig.set_title("(a) Sinal Original")
    ax_orig.grid(True, alpha=0.3)

    start_idx = 2
  else:
    axes[0, 0].plot(time_axis, x, color="b", label="Sinal Original", linewidth=1.5)
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("(a) Sinal Original")
    axes[0, 0].grid(True, alpha=0.3)

    start_idx = 1

  for i, (sensing_matrix, sparsifying_matrix, algorithm, cr) in enumerate(
    cs_architectures
  ):
    label = f"{algorithm.name} + {sensing_matrix.name} + {sparsifying_matrix.name}"

    cs = CompressedSensing(sensing_matrix, sparsifying_matrix, algorithm)

    y = cs.compress(x)
    x_hat = cs.reconstruct(y)

    print(x.shape, x_hat.shape)

    prd, nmse, sndr, ssim = cs.evaluate(x, x_hat.flatten())
    print(f"Similarity score for {label}: {ssim:.4f}")

    letter = chr(ord("b") + i)

    if ncols == 2:
      subplot_idx = start_idx + i
      ax = plt.subplot(nrows, ncols, subplot_idx)
    else:
      row = i + 1
      ax = axes[row, 0]

    ax.plot(
      time_axis,
      x,
      color="b",
      alpha=0.7,
      linewidth=1,
      label="Sinal original" if i == 0 else "",
    )
    ax.plot(
      time_axis,
      x_hat,
      color="r",
      linewidth=1.5,
      label="Sinal reconstruído" if i == 0 else "",
    )
    ax.set_ylabel("Amplitude")
    ax.set_title(
      f"({letter}) {label} (CR={cr}, PRD={prd:.2f}%, NMSE={nmse:.2f} SNDR={sndr:.2f} dB)"
    )
    ax.grid(True, alpha=0.3)

  if ncols == 2:
    first_recon_ax = plt.subplot(nrows, ncols, start_idx)
    handles, labels = first_recon_ax.get_legend_handles_labels()
  else:
    handles, labels = axes[1, 0].get_legend_handles_labels()

  plt.figlegend(handles, labels, loc="upper center", ncol=2)

  plt.xlabel("Tempo (s)")
  plt.tight_layout()
  plt.subplots_adjust(top=0.92)  # Make room for legend
  plt.show()


def main() -> None:
  loader = CHBMITLoader()

  fileName = "chb06_03.edf"
  start_time_idx = 2885120
  segment_length_s = 2
  downsampled_fs = 256
  ch_name = "T7-FT9"
  signal = loader.get_signal(
    fileName, start_time_idx, segment_length_s, downsampled_fs, ch_name
  )

  N = loader.get_downsampled_n_samples(segment_length_s, downsampled_fs)
  cs_architectures = []

  CR = 8
  M = int(N / CR)
  cs_architectures: list[CS_Architecture] = [
    (
      BinaryPermutedBlockDiagonal(M, CR),
      # Gaussian(M, N, random_state=512),
      # SparseBinary(M, N, d=8, random_state=random_state),
      DCT(N),
      # CVXPBasisPursuitIndividual(),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=180),
      SPGL1BasisPursuit(max_iter=10000, tol=1e-9),
      CR,
    ),
    # (
    #   BinaryPermutedBlockDiagonal(M, CR),
    #   Gabor(N, fs=downsampled_fs, tf=2, ff=4),
    #   CVXPBasisPursuitIndividual(),
    #   CR,
    # ),
    (
      # BinaryPermutedBlockDiagonal(M, CR),
      BinaryPermutedBlockDiagonal(M, CR),
      # Gaussian(M, N, random_state=57),
      Wavelet(N, wavelet="db4", mode="periodization"),
      CVXPBasisPursuitIndividual(),
      # SPGL1BasisPursuit(max_iter=10000, tol=1e-9),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=180),
      CR,
    ),
    (
      # BinaryPermutedBlockDiagonal(M, CR),
      Gaussian(M, N, random_state=1015),
      Gabor(N, fs=downsampled_fs, tf=2, ff=4),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=180),
      SPGL1BasisPursuit(max_iter=10000, tol=1e-9),
      CR,
    ),
    (
      # BinaryPermutedBlockDiagonal(M, CR),
      Undersampled(M, N, random_state=26),
      DCT(N),
      CVXPBasisPursuitIndividual(),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=180),
      # SPGL1BasisPursuit(max_iter=10000, tol=1e-9),
      CR,
    ),
  ]

  # CR = 4
  # M = int(N / CR)
  # cs_architectures.append(
  #   (
  #     SparseBinary(M, N, d=8, random_state=random_state),
  #     DCT(N),
  #     OrthogonalMatchingPursuit(n_nonzero_coefs=25, tol=1e-9),
  #     CR,
  #   )
  # )
  # cs_architectures.append(
  #   (
  #     SparseBinary(M, N, d=8, random_state=random_state),
  #     DCT(N),
  #     SPGL1BasisPursuit(max_iter=10000, tol=1e-9),
  #     # SPGL1BasisPursuitDenoising2(max_iter=10000, sigma_factor=0.00001),
  #     CR,
  #   ),
  # )

  plot_reconstructions(
    x=signal,
    cs_architectures=cs_architectures,
    start_time_idx=start_time_idx,
    downsampled_fs=downsampled_fs,
  )


if __name__ == "__main__":
  main()
