import matplotlib.pyplot as plt
import numpy as np

from eeg_cs.db.client import SQLiteClient


def plot_sparsifying_matrices_prd(
  db_filename: str,
  sensing_matrix: str,
  algorithm: str,
  sparsifying_matrices: list[str],
  dataset: str | None = None,
  channel: str | None = None,
  compression_rate: int | None = None,
  title: str = "PRD vs Sparsifying Matrix",
) -> None:
  client = SQLiteClient(db_filename)

  # Build base query
  query = """
        SELECT sparsifying_matrix, prd 
        FROM evaluations 
        WHERE sensing_matrix = ? AND algorithm = ?
    """
  params = [sensing_matrix, algorithm]

  # Add optional filters
  if dataset:
    query += " AND dataset = ?"
    params.append(dataset)

  if channel:
    query += " AND channel = ?"
    params.append(channel)

  if compression_rate:
    query += " AND compression_rate = ?"
    params.append(compression_rate)

  # Add sparsifying matrix filter
  placeholders = ",".join(["?" for _ in sparsifying_matrices])
  query += f" AND sparsifying_matrix IN ({placeholders})"
  params.extend(sparsifying_matrices)

  # Execute query
  print(query, params)
  results = client.query_all(query, params)

  if not results:
    print("No data found for the specified parameters")
    return

  # Organize data by sparsifying matrix
  data_by_matrix = {}
  for row in results:
    spars_matrix = row["sparsifying_matrix"]
    prd = row["prd"]
    if spars_matrix not in data_by_matrix:
      data_by_matrix[spars_matrix] = []
    data_by_matrix[spars_matrix].append(prd)

  # Prepare data for plotting
  matrices = []
  prd_values = []
  prd_means = []
  prd_stds = []
  for matrix in sparsifying_matrices:
    if matrix in data_by_matrix:
      matrices.append(matrix)
      values = data_by_matrix[matrix]
      prd_values.append(values)
      prd_means.append(np.mean(values))
      prd_stds.append(np.std(values))

  print(prd_means)
  if not matrices:
    print("No data found for any of the specified sparsifying matrices")
    return

  fig, ax = plt.subplots(figsize=(12, 8))

  box_plot = ax.boxplot(prd_values, labels=matrices, patch_artist=True)

  colors = plt.cm.Set3(np.linspace(0, 1, len(matrices)))
  for patch, color in zip(box_plot["boxes"], colors, strict=False):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

  ax.scatter(
    range(1, len(matrices) + 1),
    prd_means,
    color="red",
    marker="D",
    s=50,
    zorder=3,
    label="Mean",
  )

  ax.set_xlabel("Sparsifying Matrix", fontsize=12)
  ax.set_ylabel("PRD (%)", fontsize=12)
  ax.set_title(f"{title}\n{sensing_matrix} + {algorithm}", fontsize=14)
  ax.grid(True, alpha=0.3)
  ax.legend()

  # Rotate x-axis labels if needed
  plt.xticks(rotation=45, ha="right")

  # Add statistics text
  stats_text = f"Datasets: {len(results)} evaluations"
  if dataset:
    stats_text += f"\nDataset: {dataset}"
  if channel:
    stats_text += f"\nChannel: {channel}"
  if compression_rate:
    stats_text += f"\nCR: {compression_rate}"

  ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
  )

  plt.tight_layout()
  plt.show()


def main() -> None:
  # Example usage
  sparsifying_matrices = ["DCT", "IDCT"]

  # Single comparison
  plot_sparsifying_matrices_prd(
    db_filename="eeg_cs_evaluations.db",
    sensing_matrix="BPBD",
    algorithm="BPDN",
    sparsifying_matrices=sparsifying_matrices,
    # dataset="chbmit",
    compression_rate=2,
    title="PRD Comparison Across Sparsifying Matrices",
  )


if __name__ == "__main__":
  main()
