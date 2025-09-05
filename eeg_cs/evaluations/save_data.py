import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
import numpy.typing as npt

from eeg_cs.db.client import Evaluation, SQLiteClient
from eeg_cs.models.compressed_sensing import CompressedSensing
from eeg_cs.models.loaderv2 import BCIIIILoader, BCIIVLoader, CHBMITLoader, Loader
from eeg_cs.models.reconstruction_algorithm import (
  OrthogonalMatchingPursuit,
  ReconstructionAlgorithm,
  SPGL1BasisPursuit2,
  SPGL1BasisPursuitDenoising2,
)
from eeg_cs.models.sensing_matrix import (
  BinaryPermutedBlockDiagonal,
  SensingMatrix,
)
from eeg_cs.models.sparsifying_matrix import (
  DCT,
  SparsifyingMatrix,
  Wavelet,
)


def get_sensing_matrices_dict(
  CR: int, N: int, random_state: int = 42
) -> dict[str, SensingMatrix]:
  M = int(N / CR)

  return {
    # "Gaussian": Gaussian(M, N, random_state),
    # "Bernoulli": Bernoulli(M, N, random_state),
    # "Undersampled": Undersampled(M, N, random_state),
    "BPBD": BinaryPermutedBlockDiagonal(M, CR),
    # "SB_8": SparseBinary(M, N, d=8, random_state=random_state),
  }


def get_sparsifying_matrices_dict(N: int) -> dict[str, SparsifyingMatrix]:
  return {
    "DCT": DCT(N),
    # "DST": DST(N),
    # "CS": CS(N),
    # "ICS": ICS(N),
    # "Gabor_128": Gabor(N, fs=128),
    "Wavelet": Wavelet(N, wavelet="db4"),
  }


def get_reconstruction_algorithms_dict() -> dict[str, ReconstructionAlgorithm]:
  return {
    "OMP": OrthogonalMatchingPursuit(n_nonzero_coefs=250, tol=1e-6),
    "BPD": SPGL1BasisPursuit2(max_iter=10000, tol=1e-8),
    "BPDN": SPGL1BasisPursuitDenoising2(max_iter=10000, sigma_factor=0.00001),
  }


def get_loaders_dict(
  segment_length_s: int,
  max_blocks_per_file_per_run: int,
  random_state: int = 42,
) -> dict[str, Loader]:
  return {
    "BCIIV": BCIIVLoader(segment_length_s, max_blocks_per_file_per_run, random_state),
    "CHBMIT": CHBMITLoader(segment_length_s, max_blocks_per_file_per_run, random_state),
    "BCIIII": BCIIIILoader(segment_length_s, max_blocks_per_file_per_run, random_state),
  }


@dataclass()
class Job:
  block: npt.NDArray[np.float64]
  dataset: str
  ch_names: list[str]
  file_name: str
  start_time_s: float
  sens_m_name: str
  spars_m_name: str
  alg_name: str
  cs: CompressedSensing
  CR: int
  segment_length_s: int


def process_cs(args: Job) -> list[Evaluation]:
  X = args.block
  dataset = args.dataset
  ch_names = args.ch_names
  file_name = args.file_name
  start_time_s = args.start_time_s
  sensing_matrix = args.sens_m_name
  sparsifying_matrix = args.spars_m_name
  algorithm = args.alg_name
  cs = args.cs
  compression_rate = args.CR
  segment_length_s = args.segment_length_s

  start = time.time()
  Y = cs.compress(X)
  X_hat = cs.reconstruct(Y)
  elapsed_time_s = time.time() - start

  prds, _, sndrs = CompressedSensing.evaluate(X, X_hat)
  results: list[Evaluation] = []
  for i in range(len(ch_names)):
    prd = prds[i]
    sndr = sndrs[i]
    channel = ch_names[i]

    results.append(
      (
        dataset,
        channel,
        file_name,
        start_time_s,
        sensing_matrix,
        sparsifying_matrix,
        algorithm,
        prd,
        sndr,
        elapsed_time_s,
        compression_rate,
        segment_length_s,
      )
    )

  return results


def main() -> None:
  CR = 2
  random_state = 42

  segment_length_s = 4
  max_blocks_per_file_per_run = 10
  loaders_dict = get_loaders_dict(
    segment_length_s, max_blocks_per_file_per_run, random_state
  )

  first_loader = loaders_dict.values().__iter__().__next__()
  downsampled_fs = 128
  N = first_loader.get_downsampled_n_samples(downsampled_fs)

  sensing_matrices_dict = get_sensing_matrices_dict(CR, N, random_state)
  sparsifying_matrices_dict = get_sparsifying_matrices_dict(N)
  reconstruction_algorithms_dict = get_reconstruction_algorithms_dict()

  total_number_of_blocks_per_loader = 2
  jobs: list[Job] = []
  for loader in loaders_dict.values():
    loader_block_generator = loader.blocks_generator(downsampled_fs)
    for sens_m_name, sens_m_matrix in sensing_matrices_dict.items():
      for spars_m_name, spars_m_matrix in sparsifying_matrices_dict.items():
        for alg_name, alg in reconstruction_algorithms_dict.items():
          cs = CompressedSensing(sens_m_matrix, spars_m_matrix, alg)
          for _ in range(total_number_of_blocks_per_loader):
            (block, start_time_s, file_name) = next(loader_block_generator)

            jobs.append(
              Job(
                block,
                loader.dataset,
                loader.ch_names,
                file_name,
                start_time_s,
                sens_m_name,
                spars_m_name,
                alg_name,
                cs,
                CR,
                segment_length_s,
              )
            )

    # loader.save(
    #   f"home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/eeg_cs/evaluations/TESTE/{loader.dataset}.pkl"
    # )

  client = SQLiteClient()
  client.reset()

  buffer_size = 100
  chunksize = 50
  evaluations_buffer: list[Evaluation] = []
  with Pool(processes=cpu_count()) as pool:
    for result_batch in pool.imap_unordered(process_cs, jobs, chunksize):
      evaluations_buffer.extend(result_batch)

      if len(evaluations_buffer) >= buffer_size:
        client.bulk_insert_evaluations(evaluations_buffer)
        evaluations_buffer.clear()

  if evaluations_buffer:
    client.bulk_insert_evaluations(evaluations_buffer)


if __name__ == "__main__":
  main()
