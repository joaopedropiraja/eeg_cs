from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt

from eeg_cs.models.loader import CHBMITLoader
from eeg_cs.models.sensing_matrix import (
  BinaryPermutedBlockDiagonal,
  Gaussian,
  SensingMatrix,
  SparseBinary,
)


def plot_compressed_signals(
  x: npt.NDArray[np.float64],
  sensing_matrices: Sequence[SensingMatrix],
  CR: float,
  start_time_idx: int,
  downsampled_fs: float,
  *,
  show_original: bool = True,
) -> None:
  segment_length = len(x)
  start_time_s = start_time_idx / downsampled_fs
  time_axis = start_time_s + np.arange(segment_length) / downsampled_fs

  n_matrices = len(sensing_matrices)

  nrows = n_matrices + 1 if show_original else n_matrices
  ncols = 1

  fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows), sharex=True)

  if nrows == 1:
    axes = np.array([axes])

  plot_idx = 0
  if show_original:
    letter = chr(ord("a") + plot_idx)
    axes[plot_idx].plot(time_axis, x, color="b", linewidth=1.5)
    axes[plot_idx].grid(visible=True, alpha=0.3)

    # Force scientific notation on y-axis
    axes[plot_idx].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axes[plot_idx].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Add letter to the right side
    axes[plot_idx].text(
      1.02,
      0.5,
      f"({letter})",
      transform=axes[plot_idx].transAxes,
      fontsize=12,
      fontweight="bold",
      va="center",
    )
    plot_idx += 1

  for _, sensing_matrix in enumerate(sensing_matrices):
    y = sensing_matrix.value @ x

    compressed_time_axis = start_time_s + np.arange(len(y)) / (downsampled_fs / CR)

    letter = chr(ord("a") + plot_idx)
    ax = axes[plot_idx]

    ax.plot(
      compressed_time_axis,
      y,
      color="b",
      linewidth=1.5,
      markersize=3,
      label=f"Comprimido ({sensing_matrix.name})",
    )
    # ax.set_title(
    #   f"({letter}) Sinal Comprimido - {sensing_matrix.name} "
    #   f"(CR={CR}, M={len(y)}, N={segment_length})",
    #   fontsize=11,
    #   fontweight="bold",
    # )
    ax.grid(visible=True, alpha=0.3)
    # ax.legend(loc="upper right")

    # Force scientific notation on y-axis
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Add letter to the right side
    ax.text(
      1.02,
      0.5,
      f"({letter})",
      transform=ax.transAxes,
      fontsize=12,
      fontweight="bold",
      va="center",
    )

    # Print compression info
    print(f"{sensing_matrix.name}: N={segment_length} -> M={len(y)} (CR={CR})")

    plot_idx += 1

  # Set common x-label
  axes[-1].set_xlabel("Tempo (s)", fontsize=11)

  # Add common y-label for all subplots
  fig.supylabel("Amplitude", fontsize=11, x=0.02)

  # plt.suptitle(
  #   "Comparação de Sinais Comprimidos",
  #   fontsize=13,
  #   fontweight="bold",
  #   y=0.995,
  # )
  plt.tight_layout()
  plt.savefig("compressed_signals.png", dpi=500)
  # plt.suptitle(
  plt.show()


def main() -> None:
  loader = CHBMITLoader()

  file_name = "chb02_02.edf"
  start_time_idx = 0
  segment_length_s = 1
  downsampled_fs = 256
  ch_name = "F3-C3"

  signal = loader.get_signal(
    file_name, start_time_idx, segment_length_s, downsampled_fs, ch_name
  )

  n_samples = loader.get_downsampled_n_samples(segment_length_s, downsampled_fs)

  CR = 2
  m_measurements = int(n_samples / CR)

  random_state = 1015
  sensing_matrices = [
    Gaussian(m_measurements, n_samples, random_state=random_state),
    SparseBinary(m_measurements, n_samples, d=16, random_state=random_state),
    BinaryPermutedBlockDiagonal(m_measurements, CR),
  ]

  plot_compressed_signals(
    x=signal,
    sensing_matrices=sensing_matrices,
    CR=CR,
    start_time_idx=start_time_idx,
    downsampled_fs=downsampled_fs,
    show_original=True,
  )


if __name__ == "__main__":
  main()
