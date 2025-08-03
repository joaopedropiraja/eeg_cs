import matplotlib.pyplot as plt
import numpy as np
from cr.sparse import lop
from cr.sparse.cvx.spgl1 import SPGL1Options, solve_bp
from scipy.fftpack import dct

from eeg_cs.models.loader import CHBMITLoader


def plot_reconstructed_signals(
  original_data: np.ndarray, reconstructed_data: np.ndarray, n_channels=3
):
  """
  Plot original and reconstructed signals for a few channels.

  Parameters
  ----------
  original_data : np.ndarray
      Original signals (shape: n_blocks, n_samples, n_channels).
  reconstructed_data : np.ndarray
      Reconstructed signals (shape: n_blocks, n_samples, n_channels).
  n_channels : int
      Number of channels to plot.
  """
  n_blocks, n_samples, n_channels_total = original_data.shape
  channels_to_plot = min(n_channels, n_channels_total)

  for block_idx in range(n_blocks):
    plt.figure(figsize=(12, 6))
    for channel_idx in range(channels_to_plot):
      plt.subplot(channels_to_plot, 1, channel_idx + 1)
      plt.plot(original_data[block_idx, :, channel_idx], label="Original", alpha=0.7)
      plt.plot(
        reconstructed_data[block_idx, :, channel_idx], label="Reconstructed", alpha=0.7
      )
      plt.title(f"Block {block_idx + 1}, Channel {channel_idx + 1}")
      plt.legend()
      plt.grid()
    plt.tight_layout()
    plt.show()


def compressed_sensing_pipeline(
  loader: CHBMITLoader, CR: int = 2, random_state: int = 42
) -> None:
  """
  Compressed sensing pipeline for CHBMITLoader using cr.sparse.

  Parameters
  ----------
  loader : CHBMITLoader
      Loader for CHB-MIT dataset.
  CR : int
      Compression ratio (M/N).
  random_state : int
      Random seed for reproducibility.

  Returns
  -------
  None
  """
  # Step 1: Load data
  data = loader.data  # Shape: (n_blocks, n_samples, n_channels)
  n_blocks, N, n_channels = data.shape

  # Step 2: Define sensing matrix
  M = int(N / CR)  # Number of measurements
  rng = np.random.default_rng(random_state)
  sensing_matrix = rng.choice([-1, 1], size=(M, N))  # Bernoulli sensing matrix
  sensing_operator = lop.matrix(sensing_matrix)

  # Step 3: Define sparsifying basis (DCT)
  dct_basis = dct(np.eye(N), norm="ortho")  # DCT basis matrix

  # Step 4: Compress data
  compressed_data = []
  for block in data:
    compressed_block = sensing_matrix @ block  # Shape: (M, n_channels)
    compressed_data.append(compressed_block)
  compressed_data = np.array(compressed_data)  # Shape: (n_blocks, M, n_channels)

  # Step 5: Reconstruct data
  reconstructed_data = []
  options = SPGL1Options(max_iters=1000)
  for block in compressed_data:
    reconstructed_block = []
    for channel in block.T:
      # Solve Basis Pursuit problem
      solution = solve_bp(sensing_operator, channel, options=options)
      sparse_representation = solution.x  # Sparse coefficients
      reconstructed_channel = dct_basis @ sparse_representation  # Reconstruct signal
      reconstructed_block.append(reconstructed_channel)
    reconstructed_data.append(np.array(reconstructed_block).T)
  reconstructed_data = np.array(reconstructed_data)  # Shape: (n_blocks, N, n_channels)

  # Step 6: Evaluate reconstruction
  for i, (original_block, reconstructed_block) in enumerate(
    zip(data, reconstructed_data, strict=False)
  ):
    prd = (
      100
      * np.linalg.norm(original_block - reconstructed_block)
      / np.linalg.norm(original_block)
    )
    print(f"Block {i + 1}: PRD = {prd:.2f}%")

  plot_reconstructed_signals(data, reconstructed_data, n_channels=min(3, n_channels))


if __name__ == "__main__":
  # Initialize loader
  loader = CHBMITLoader(n_blocks=5, segment_length_sec=4)
  loader.downsample(new_fs=128)  # Downsample to 128 Hz

  # Run compressed sensing pipeline
  compressed_sensing_pipeline(loader, CR=2, random_state=42)
