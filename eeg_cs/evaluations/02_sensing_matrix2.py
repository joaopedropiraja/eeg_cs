import time
from multiprocessing import Pool, cpu_count
from os import path

import duckdb
import pandas as pd

from eeg_cs.models.compressed_sensing import CompressedSensing
from eeg_cs.models.loader2 import BCIIIILoader, BCIIVLoader, CHBMITLoader, Loader
from eeg_cs.models.reconstruction_algorithm import (
  OrthogonalMatchingPursuit,
  SPGL1BasisPursuit2,
  SPGL1BasisPursuitDenoising2,
)
from eeg_cs.models.sensing_matrix import (
  Bernoulli,
  BinaryPermutedBlockDiagonal,
  Gaussian,
  SparseBinary,
  Undersampled,
)
from eeg_cs.models.sparsifying_matrix import DCT


def _process_sensing_matrix(args) -> list[dict]:
  """
  Helper function for multiprocessing.
  Processes a single sensing matrix and reconstruction algorithm combination.
  """
  dataset, ch_names, sensing_name, alg_name, cs, X = args
  start = time.time()
  Y = cs.compress(X)
  X_hat = cs.reconstruct(Y)
  elapsed = time.time() - start

  prd, nmse, sndr = CompressedSensing.evaluate(X, X_hat)
  results = []
  for i in range(prd.size):
    prd_val = prd[i]
    nmse_val = nmse[i]
    sndr_val = sndr[i]
    ch_name = ch_names[i]

    results.append(
      {
        "Dataset": dataset,
        "Channel": ch_name,
        "Sensing Matrix": sensing_name,
        "Algorithm": alg_name,
        "PRD": prd_val,
        "NMSE": nmse_val,
        "SNDR": sndr_val,
        "Time (s)": elapsed,
      }
    )
  return results


def compare_sensing_matrices(
  loaders: list[Loader],
  CR: int = 2,
  folder: str = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/eeg_cs/evaluations/sensing_matrix",
  random_state: int = 42,
) -> None:
  n_blocks, N, _ = loaders[0].data.shape  # (n_blocks, n_samples, n_channels)
  M = int(N / CR)

  # Use DCT sparse basis
  dct_basis = DCT(N)

  # Sensing matrices
  sensing_matrices = {
    "Gaussian": Gaussian(M, N, random_state),
    "Bernoulli": Bernoulli(M, N, random_state),
    "Undersampled": Undersampled(M, N, random_state),
    "BPBD": BinaryPermutedBlockDiagonal(M, CR),
    "SB": SparseBinary(M, N, d=8, random_state=random_state),
  }

  # Reconstruction algorithms
  algorithms = {
    "OMP": OrthogonalMatchingPursuit(n_nonzero_coefs=250, tol=1e-8),
    "BPD": SPGL1BasisPursuit2(tol=1e-8, max_iter=10000),
    "BPDN": SPGL1BasisPursuitDenoising2(sigma_factor=0.0001, max_iter=10000),
  }

  jobs = []
  for sensing_name, sensing in sensing_matrices.items():
    for alg_name, alg in algorithms.items():
      cs = CompressedSensing(sensing, dct_basis, alg)
      for loader in loaders:
        for X in loader.data:
          jobs.append((loader.dataset, loader.ch_names, sensing_name, alg_name, cs, X))

  # Use multiprocessing to parallelize the jobs
  with Pool(processes=cpu_count()) as pool:
    results_nested = pool.map(_process_sensing_matrix, jobs)

  # Flatten results
  results = [item for sublist in results_nested for item in sublist]

  file_name = f"m_{M}_cr_{CR}_nblocks_{n_blocks}_rs_{random_state}"
  df = pd.DataFrame(results)
  df.to_csv(path.join(folder, f"{file_name}.csv"), index=False)


def sensing_matrix_test() -> None:
  n_blocks = 10
  segment_length_sec = 4
  max_blocks_per_file_per_run = 2
  fs = 128
  CR = 4

  random_state = 20

  loader1 = CHBMITLoader(n_blocks, segment_length_sec, max_blocks_per_file_per_run)
  loader2 = BCIIVLoader(n_blocks, segment_length_sec, max_blocks_per_file_per_run)
  loader3 = BCIIIILoader(n_blocks, segment_length_sec, max_blocks_per_file_per_run)

  loaders = [loader1, loader2, loader3]
  for loader in loaders:
    loader.downsample(new_fs=fs)

  start = time.time_ns()

  compare_sensing_matrices(loaders, CR, random_state=random_state)

  print(f"Total execution time: {(time.time_ns() - start) / 1e9}")


def agg_results(csv_path: str) -> None:
  con = duckdb.connect(database=":memory:")
  con.execute(f"CREATE TABLE results AS SELECT * FROM read_csv_auto('{csv_path}')")

  query = """
    SELECT
        "Sensing Matrix",
        "Algorithm",
        AVG("PRD") AS mean_PRD,
        AVG("NMSE") AS mean_NMSE,
        AVG("SNDR") AS mean_SNDR,
        AVG("Time (s)") AS mean_Time_s
    FROM results
    GROUP BY "Sensing Matrix", "Algorithm"
    ORDER BY "Sensing Matrix", "Algorithm"
    """

  agg_df = con.execute(query).df()
  print(agg_df)

  # Optionally, save the aggregated results
  agg_df.to_csv(f"{csv_path.split('.')[0]}_duckdb_agg.csv", index=False)


if __name__ == "__main__":
  # sensing_matrix_test()

  csv_path = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/eeg_cs/evaluations/sensing_matrix/m_128_cr_4_nblocks_10_rs_20.csv"
  agg_results(csv_path)
