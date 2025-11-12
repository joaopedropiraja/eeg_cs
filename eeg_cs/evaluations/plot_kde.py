from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from eeg_cs.db.client import SQLiteClient


def plot_metric_kde(
  db_filename: str,
  metric: str,
  sensing_matrix: str | None = None,
  sparsifying_matrix: str | None = None,
  algorithm: str | None = None,
  dataset: str | None = None,
  channel: str | None = None,
  compression_rate: int | None = None,
  segment_length_s: int | None = None,
  title: str | None = None,
  figsize: tuple[int, int] = (12, 6),
  color: str = "steelblue",
  fill: bool = True,
  alpha: float = 0.6,
  show_stats: bool = True,
) -> None:
  client = SQLiteClient(db_filename)

  query = f"SELECT {metric} FROM evaluations WHERE 1=1"
  params: list[str | int] = []

  if sensing_matrix is not None:
    query += " AND sensing_matrix = ?"
    params.append(sensing_matrix)

  if sparsifying_matrix is not None:
    query += " AND sparsifying_matrix = ?"
    params.append(sparsifying_matrix)

  if algorithm is not None:
    query += " AND algorithm = ?"
    params.append(algorithm)

  if dataset is not None:
    query += " AND dataset = ?"
    params.append(dataset)

  if channel is not None:
    query += " AND channel = ?"
    params.append(channel)

  if compression_rate is not None:
    query += " AND compression_rate = ?"
    params.append(compression_rate)

  if segment_length_s is not None:
    query += " AND segment_length_s = ?"
    params.append(segment_length_s)

  results = client.query_all(query, params)

  if not results:
    print("No data found for the specified filters")
    return

  values = np.array([row[metric] for row in results])

  mean_val = np.mean(values)
  std_val = np.std(values)
  median_val = np.median(values)
  min_val = np.min(values)
  max_val = np.max(values)

  fig, ax = plt.subplots(figsize=figsize)

  sns.kdeplot(data=values, ax=ax, color=color, fill=fill, alpha=alpha, linewidth=2)

  ax.axvline(
    mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.4f}"
  )
  ax.axvline(
    median_val,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {median_val:.4f}",
  )

  metric_labels = {
    "prd": "PRD (%)",
    "nmse": "NMSE",
    "sndr": "SNDR (dB)",
    "ssim": "SSIM",
    "elapsed_time_s": "Elapsed Time (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  ax.set_xlabel(metric_labels.get(metric, metric.upper()), fontsize=12)
  ax.set_ylabel("Density", fontsize=12)

  if title is None:
    filter_parts = []
    if sensing_matrix:
      filter_parts.append(f"Sensing: {sensing_matrix}")
    if sparsifying_matrix:
      filter_parts.append(f"Sparsifying: {sparsifying_matrix}")
    if algorithm:
      filter_parts.append(f"Algorithm: {algorithm}")
    if dataset:
      filter_parts.append(f"Dataset: {dataset}")
    if compression_rate:
      filter_parts.append(f"CR: {compression_rate}")

    filter_str = " | ".join(filter_parts) if filter_parts else "All Data"
    title = f"KDE of {metric_labels.get(metric, metric.upper())}\n{filter_str}"

  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=10)

  if show_stats:
    stats_text = (
      f"n = {len(values)}\n"
      f"Mean: {mean_val:.4f}\n"
      f"Std: {std_val:.4f}\n"
      f"Median: {median_val:.4f}\n"
      f"Min: {min_val:.4f}\n"
      f"Max: {max_val:.4f}"
    )

    ax.text(
      0.98,
      0.98,
      stats_text,
      transform=ax.transAxes,
      verticalalignment="top",
      horizontalalignment="right",
      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
      fontsize=9,
      fontfamily="monospace",
    )

  plt.tight_layout()
  plt.show()


def plot_metric_kde_comparison(
  db_filename: str,
  metric: str,
  group_by: str,
  group_values: list[str],
  sensing_matrix: str | None = None,
  sparsifying_matrix: str | None = None,
  algorithm: str | None = None,
  dataset: str | None = None,
  channel: str | None = None,
  compression_rate: int | None = None,
  segment_length_s: int | None = None,
  title: str | None = None,
  figsize: tuple[int, int] = (14, 6),
  colors: list[str] | None = None,
  fill: bool = True,
  alpha: float = 0.4,
  show_legend: bool = True,
) -> None:
  client = SQLiteClient(db_filename)

  if colors is None:
    colors = plt.cm.Set2(np.linspace(0, 1, len(group_values)))

  fig, ax = plt.subplots(figsize=figsize)

  all_stats = []

  for i, group_value in enumerate(group_values):
    query = f"SELECT {metric} FROM evaluations WHERE {group_by} = ?"
    params: list[str | int] = [group_value]

    if sensing_matrix is not None and group_by != "sensing_matrix":
      query += " AND sensing_matrix = ?"
      params.append(sensing_matrix)

    if sparsifying_matrix is not None and group_by != "sparsifying_matrix":
      query += " AND sparsifying_matrix = ?"
      params.append(sparsifying_matrix)

    if algorithm is not None and group_by != "algorithm":
      query += " AND algorithm = ?"
      params.append(algorithm)

    if dataset is not None and group_by != "dataset":
      query += " AND dataset = ?"
      params.append(dataset)

    if channel is not None and group_by != "channel":
      query += " AND channel = ?"
      params.append(channel)

    if compression_rate is not None and group_by != "compression_rate":
      query += " AND compression_rate = ?"
      params.append(compression_rate)

    if segment_length_s is not None and group_by != "segment_length_s":
      query += " AND segment_length_s = ?"
      params.append(segment_length_s)

    results = client.query_all(query, params)

    if not results:
      print(f"No data found for {group_by} = {group_value}")
      continue

    values = np.array([row[metric] for row in results])

    mean_val = np.mean(values)
    std_val = np.std(values)

    sns.kdeplot(
      data=values,
      ax=ax,
      color=colors[i],
      fill=fill,
      alpha=alpha,
      linewidth=2,
      label=f"{group_value} (μ={mean_val:.3f}, σ={std_val:.3f})",
    )

    all_stats.append(
      {
        "group": group_value,
        "mean": mean_val,
        "std": std_val,
        "median": np.median(values),
        "n": len(values),
      }
    )

  metric_labels = {
    "prd": "PRD (%)",
    "nmse": "NMSE",
    "sndr": "SNDR (dB)",
    "ssim": "SSIM",
    "elapsed_time_s": "Elapsed Time (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  ax.set_xlabel(metric_labels.get(metric, metric.upper()), fontsize=12)
  ax.set_ylabel("Density", fontsize=12)

  if title is None:
    filter_parts = []
    if sensing_matrix and group_by != "sensing_matrix":
      filter_parts.append(f"Sensing: {sensing_matrix}")
    if sparsifying_matrix and group_by != "sparsifying_matrix":
      filter_parts.append(f"Sparsifying: {sparsifying_matrix}")
    if algorithm and group_by != "algorithm":
      filter_parts.append(f"Algorithm: {algorithm}")
    if dataset and group_by != "dataset":
      filter_parts.append(f"Dataset: {dataset}")
    if compression_rate and group_by != "compression_rate":
      filter_parts.append(f"CR: {compression_rate}")

    filter_str = " | ".join(filter_parts) if filter_parts else ""
    title = f"KDE Comparison of {metric_labels.get(metric, metric.upper())} by {group_by.replace('_', ' ').title()}"
    if filter_str:
      title += f"\n{filter_str}"

  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, alpha=0.3)

  if show_legend:
    ax.legend(fontsize=9, loc="best")

  plt.tight_layout()
  plt.show()

  print(f"\n{'=' * 70}")
  print(f"Statistics Summary by {group_by}:")
  print(f"{'=' * 70}")
  for stat in all_stats:
    print(
      f"{stat['group']:20s} | n={stat['n']:5d} | "
      f"mean={stat['mean']:8.4f} | std={stat['std']:8.4f} | "
      f"median={stat['median']:8.4f}"
    )
  print(f"{'=' * 70}\n")


def main() -> None:
  plot_metric_kde(
    db_filename="eeg_cs_evaluations.db",
    dataset="chbmit",
    metric="nmse",
    sensing_matrix="BPBD",
    sparsifying_matrix="Wavelet_db4_5L",
    algorithm="CVXP_BP_IND_CLARABEL",
    compression_rate=2,
  )

  # plot_metric_kde_comparison(
  #   metric="nmse",
  #   group_by="sparsifying_matrix",
  #   group_values=["DCT", "IDCT"],
  #   sensing_matrix="BPBD",
  #   algorithm="BPDN",
  #   compression_rate=2,
  # )


if __name__ == "__main__":
  main()
