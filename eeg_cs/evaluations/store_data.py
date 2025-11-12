import logging
import time
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
import numpy.typing as npt
from scipy.signal import welch

from eeg_cs.db.client import Evaluation, SQLiteClient
from eeg_cs.models.compressed_sensing import CompressedSensing
from eeg_cs.models.loader import BCIIIILoader, BCIIVLoader, CHBMITLoader, Loader
from eeg_cs.models.reconstruction_algorithm import (
  CVXPBasisPursuitIndividual,
  ReconstructionAlgorithm,
)
from eeg_cs.models.sensing_matrix import (
  BinaryPermutedBlockDiagonal,
  SensingMatrix,
)
from eeg_cs.models.sparsifying_matrix import SparsifyingMatrix, Wavelet

# Suppress all warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_mean_frequency(
  segments: npt.NDArray[np.float64], fs: int
) -> npt.NDArray[np.float64]:
  """
  Calculates the mean frequency (spectral mean) of signal segments.

  Args:
      segments (np.array): Signal data. Can be:
                          - 1D array: single segment
                          - 2D array: (n_samples, n_channels) for multi-channel data
      fs (int): The sampling frequency of the signal in Hz.

  Returns:
      np.ndarray: The mean frequency of each segment/channel in Hz.
                 - For 1D input: returns 1D array with single value
                 - For 2D input: returns 1D array with one value per channel
  """
  segments = np.asarray(segments)

  # Handle different input shapes
  if segments.ndim == 1:
    # Single segment case
    segments = segments.reshape(-1, 1)
    single_segment = True
  elif segments.ndim == 2:
    # Multi-channel case: (n_samples, n_channels)
    single_segment = False
  else:
    error_msg = f"Input must be 1D or 2D array, got {segments.ndim}D"
    raise ValueError(error_msg)

  _, n_channels = segments.shape
  mean_frequencies = np.zeros(n_channels, dtype=np.float64)

  for ch in range(n_channels):
    segment = segments[:, ch]

    # Calculate the Power Spectral Density (PSD) using Welch's method.
    # 'f' will be an array of frequencies.
    # 'Pxx' will be an array of power spectral densities for those frequencies.
    # We set nperseg to the length of the segment to get a high-resolution
    # periodogram, similar to what's implied by the paper.
    f, Pxx = welch(segment, fs=fs, nperseg=len(segment))

    # Calculate the mean frequency.
    # This is a weighted average of the frequencies (f),
    # where the weights are the power values (Pxx).
    mean_frequencies[ch] = np.average(f, weights=Pxx)

  return mean_frequencies if not single_segment else mean_frequencies[0]


@dataclass()
class Job:
  block: npt.NDArray[np.float64]
  dataset: str
  fs: int
  ch_names: list[str]
  file_name: str
  start_time_idx: int
  sens_m_name: str
  spars_m_name: str
  alg_name: str
  cs: CompressedSensing
  CR: int
  segment_length_s: int


def execute_job(job: Job) -> list[Evaluation]:
  X = job.block
  dataset = job.dataset
  fs = job.fs
  ch_names = job.ch_names
  file_name = job.file_name
  start_time_idx = job.start_time_idx
  sensing_matrix = job.sens_m_name
  sparsifying_matrix = job.spars_m_name
  algorithm = job.alg_name
  cs = job.cs
  compression_rate = job.CR
  segment_length_s = job.segment_length_s

  try:
    # Calculate mean frequencies before any processing that might modify X
    mean_frequencies = calculate_mean_frequency(X, fs)

    start = time.time()
    Y = cs.compress(X)
    X_hat = cs.reconstruct(Y)
    elapsed_time_s = time.time() - start

    del Y

    prds, nmses, sndrs, ssims = CompressedSensing.evaluate(X, X_hat)

    del X, X_hat

    results: list[Evaluation] = []
    for i, channel in enumerate(ch_names):
      results.append(
        (
          dataset,
          fs,
          channel,
          file_name,
          start_time_idx,
          sensing_matrix,
          sparsifying_matrix,
          algorithm,
          float(mean_frequencies[i]),
          float(prds[i]),
          float(nmses[i]),
          float(sndrs[i]),
          float(ssims[i]),
          elapsed_time_s,
          compression_rate,
          segment_length_s,
        )
      )

    print(
      f"Processed {dataset} | File: {file_name} | Start idx: {start_time_idx} | "
      f"Sens: {sensing_matrix} | Spars: {sparsifying_matrix} | Alg: {algorithm} | "
      f"CR: {compression_rate} | PRD: {np.mean(prds):.4f} | NMSE: {np.mean(nmses):.6f} | "
      f"SNDR: {np.mean(sndrs):.2f} dB | Time: {elapsed_time_s:.2f} s"
    )

  except Exception:
    logger.exception(f"Error processing job for {dataset}/{file_name}")
    return []
  else:
    return results


def create_jobs_generator(
  loaders: list[Loader],
  sensing_matrices: list[SensingMatrix],
  sparsifying_matrices: list[SparsifyingMatrix],
  reconstruction_algorithms: list[ReconstructionAlgorithm],
  total_number_of_blocks_per_loader: int,
  downsampled_fs: int,
  CR: int,
  segment_length_s: int,
  max_blocks_per_file_per_run: int | None = None,
  random_state: int | None = None,
) -> Generator[Job, None, None]:
  for loader in loaders:
    loader_block_generator = loader.blocks_generator(
      segment_length_s,
      max_blocks_per_file_per_run,
      downsampled_fs,
      random_state,
    )
    for _ in range(total_number_of_blocks_per_loader):
      try:
        block, start_time_idx, file_name = next(loader_block_generator)
        for sensing_matrix in sensing_matrices:
          for sparsifying_matrix in sparsifying_matrices:
            for algorithm in reconstruction_algorithms:
              cs = CompressedSensing(sensing_matrix, sparsifying_matrix, algorithm)
              yield Job(
                block,
                loader.dataset,
                downsampled_fs if downsampled_fs else int(loader.fs),
                loader.ch_names,
                file_name,
                start_time_idx,
                sensing_matrix.name,
                sparsifying_matrix.name,
                algorithm.name,
                cs,
                CR,
                segment_length_s,
              )
      except StopIteration:
        logger.warning(f"Ran out of blocks for {loader.dataset}")
        break


def estimate_total_jobs(
  loaders_dict: list[Loader],
  sensing_matrices: list[SensingMatrix],
  sparsifying_matrices: list[SparsifyingMatrix],
  reconstruction_algorithms: list[ReconstructionAlgorithm],
  total_number_of_blocks_per_loader: int,
) -> int:
  return (
    len(loaders_dict)
    * len(sensing_matrices)
    * len(sparsifying_matrices)
    * len(reconstruction_algorithms)
    * total_number_of_blocks_per_loader
  )


def main() -> None:
  client = SQLiteClient(db_filename="eeg_cs_evaluations.db")
  # client.reset()

  CRs = [2]
  for CR in CRs:
    logger.info(f"Starting evaluations for Compression Rate (CR): {CR}")

    segment_length_s = 4
    total_number_of_blocks_per_loader = 300
    downsampled_fs = 128
    max_blocks_per_file_per_run = 5

    buffer_size = 1000  # Larger buffer for better DB performance
    chunksize = 25  # Smaller chunks for better memory management

    logger.info("Initializing loaders and matrices...")

    loaders: list[Loader] = [CHBMITLoader(), BCIIVLoader(), BCIIIILoader()]

    first_loader = loaders[0]
    N = first_loader.get_downsampled_n_samples(segment_length_s, downsampled_fs)
    M = N // CR

    logger.info(f"Signal length (N): {N}, Measurements (M): {M}")

    sensing_matrices: list[SensingMatrix] = [
      # Gaussian(M, N, random_state=512),
      # SparseBinary(M, N, d=8, random_state=338),
      # SparseBinary(M, N, d=12, random_state=234),
      # SparseBinary(M, N, d=16, random_state=426),
      # SparseBinary(M, N, d=20, random_state=50),
      # SparseBinary(M, N, d=24, random_state=489),
      # SparseBinary(M, N, d=28, random_state=639),
      # SparseBinary(M, N, d=32, random_state=418),
      # SparseBinary(M, N, d=36, random_state=926),
      # SparseBinary(M, N, d=40, random_state=185),
      BinaryPermutedBlockDiagonal(M, CR),
    ]
    sparsifying_matrices: list[SparsifyingMatrix] = [
      # DCT(N),
      Wavelet(N, wavelet="db4", mode="periodization", levels=5),
      # Wavelet(N, wavelet="sym6", mode="periodization", levels=5),
      # DST(N),
      # CS(N),
      # IDCT(N),
      # ICS(N),
    ]

    reconstruction_algorithms: list[ReconstructionAlgorithm] = [
      # Cosamp(sparsity=100, max_iter=10000, tol=1e-9),
      # SPGL1BasisPursuit(max_iter=10000000, tol=1e-9),
      # SPGL1BasisPursuitDenoising(max_iter=10000000, sigma_factor=1e-9),
      # CVXPBasisPursuitDenoisingIndividual(sigma_factor=1e-6),
      # CVXPBasisPursuit(),
      CVXPBasisPursuitIndividual(),
      # OrthogonalMatchingPursuit(tol=1e-12),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=10),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=20),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=30),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=40),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=50),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=60),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=80),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=100),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=120),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=150),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=180),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=210),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=100),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=150),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=180),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=30, tol=1e-9),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=35, tol=1e-9),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=40, tol=1e-9),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=45, tol=1e-9),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=50, tol=1e-9),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=75, tol=1e-9),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=100, tol=1e-8),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=150, tol=1e-8),
      # OrthogonalMatchingPursuit(n_nonzero_coefs=250, tol=1e-8),
    ]

    # Estimate total for progress tracking
    total_estimated = estimate_total_jobs(
      loaders,
      sensing_matrices,
      sparsifying_matrices,
      reconstruction_algorithms,
      total_number_of_blocks_per_loader,
    )

    logger.info(f"Estimated total jobs: {total_estimated}")

    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Using {num_processes} processes")

    jobs_generator = create_jobs_generator(
      loaders,
      sensing_matrices,
      sparsifying_matrices,
      reconstruction_algorithms,
      total_number_of_blocks_per_loader,
      downsampled_fs,
      CR,
      segment_length_s,
      max_blocks_per_file_per_run,
      random_state=42,
    )

    evaluations_buffer: list[Evaluation] = []
    processed_count = 0

    with Pool(processes=num_processes) as pool:
      for result_batch in pool.imap_unordered(execute_job, jobs_generator, chunksize):
        if result_batch:
          evaluations_buffer.extend(result_batch)
          processed_count += 1

        if len(evaluations_buffer) >= buffer_size:
          rows_inserted = client.bulk_insert_evaluations(evaluations_buffer)
          logger.info(
            f"Inserted {rows_inserted} evaluations. "
            f"Progress: {processed_count}/{total_estimated}"
          )
          evaluations_buffer.clear()

        if processed_count % 50 == 0:
          logger.info(f"Processed {processed_count}/{total_estimated} jobs")

    if evaluations_buffer:
      rows_inserted = client.bulk_insert_evaluations(evaluations_buffer)
      logger.info(f"Final insert: {rows_inserted} evaluations")

    logger.info(f"Processing complete! Total jobs processed: {processed_count}")

    # for loader in loaders:
    #   loader.save(f"temp/{loader.dataset}_state.pkl")


if __name__ == "__main__":
  main()
