from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from eeg_cs.db.client import SQLiteClient


def count_outliers(
  data: npt.NDArray[np.float64],
) -> tuple[int, npt.NDArray[np.float64]]:
  """Count the number of outliers using the IQR method.

  Args:
      data: Array of data values

  Returns:
      Tuple of (number of outliers, array of outlier values)
  """
  q1 = np.percentile(data, 25)
  q3 = np.percentile(data, 75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr
  outliers = data[(data < lower_bound) | (data > upper_bound)]
  return len(outliers), outliers


def get_outlier_count(
  db_filename: str,
  metric: str,
  sensing_matrix: str | None = None,
  sparsifying_matrix: str | None = None,
  algorithm: str | None = None,
  dataset: str | None = None,
  channel: str | None = None,
  compression_rate: int | None = None,
  segment_length_s: int | None = None,
) -> tuple[npt.NDArray[np.float64], int, int, float]:
  """Get outlier count for a specific metric with optional filters.

  Args:
      db_filename: Path to the SQLite database file
      metric: Metric to analyze (prd, nmse, sndr, ssim, elapsed_time_s, mean_freq)
      sensing_matrix: Filter by sensing matrix name
      sparsifying_matrix: Filter by sparsifying matrix name
      algorithm: Filter by algorithm name
      dataset: Filter by dataset name
      channel: Filter by channel name
      compression_rate: Filter by compression rate
      segment_length_s: Filter by segment length in seconds

  Returns:
      Tuple of (number of outliers, total count, outlier percentage)
  """
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
    return 0, 0, 0.0

  values = np.array([row[metric] for row in results])
  n_outliers, outliers = count_outliers(values)
  total = len(values)
  percentage = (n_outliers / total) * 100 if total > 0 else 0.0

  return outliers, n_outliers, total, percentage


def plot_metric_boxplot(
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
  figsize: tuple[int, int] = (8, 6),
  color: str = "steelblue",
  show_stats: bool = True,
  show_outliers: bool = True,
  remove_outliers: bool = False,
  ylim_percentile: float | None = None,
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

  # Store original count for reporting
  original_count = len(values)

  # Remove outliers if requested (before calculating statistics)
  if remove_outliers:
    q1_temp = np.percentile(values, 25)
    q3_temp = np.percentile(values, 75)
    iqr_temp = q3_temp - q1_temp
    lower_bound_temp = q1_temp - 1.5 * iqr_temp
    upper_bound_temp = q3_temp + 1.5 * iqr_temp
    outlier_mask = (values >= lower_bound_temp) & (values <= upper_bound_temp)
    values = values[outlier_mask]
    removed_count = original_count - len(values)
    print(
      f"Removed {removed_count} outliers ({removed_count / original_count * 100:.2f}%)"
    )

  mean_val = np.mean(values)
  std_val = np.std(values)
  median_val = np.median(values)
  q1 = np.percentile(values, 25)
  q3 = np.percentile(values, 75)
  iqr = q3 - q1
  min_val = np.min(values)
  max_val = np.max(values)

  # Count outliers (from the current data, after potential removal)
  n_outliers, _ = count_outliers(values)
  outlier_percentage = (n_outliers / len(values)) * 100

  fig, ax = plt.subplots(figsize=figsize)

  box_parts = ax.boxplot(
    values,
    vert=True,
    patch_artist=True,
    showfliers=show_outliers,
    widths=0.5,
    boxprops=dict(facecolor=color, alpha=0.6, linewidth=1.5),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(
      marker="o",
      markerfacecolor="red",
      markersize=5,
      alpha=0.5,
      linestyle="none",
    ),
  )

  # Add mean marker
  ax.plot(1, mean_val, marker="D", color="green", markersize=8, zorder=3)

  metric_labels = {
    "prd": "PRD (%)",
    "nmse": "NMSE",
    "sndr": "SNDR (dB)",
    "ssim": "SSIM",
    "elapsed_time_s": "Elapsed Time (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)
  ax.set_xticks([])

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
    title = f"Boxplot of {metric_labels.get(metric, metric.upper())}\n{filter_str}"

  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, alpha=0.3, axis="y")

  # Add text labels on the right side
  ax.text(
    1.02,
    mean_val,
    f"Mean: {mean_val:.4f}",
    transform=ax.get_yaxis_transform(),
    verticalalignment="center",
    fontsize=8,
    color="green",
  )
  ax.text(
    1.02,
    q1,
    f"Q1: {q1:.4f}",
    transform=ax.get_yaxis_transform(),
    verticalalignment="center",
    fontsize=8,
    color="blue",
  )
  ax.text(
    1.02,
    q3,
    f"Q3: {q3:.4f}",
    transform=ax.get_yaxis_transform(),
    verticalalignment="center",
    fontsize=8,
    color="blue",
  )

  # Set y-axis limits to focus on main distribution if specified
  if ylim_percentile is not None and show_outliers:
    y_min = np.min(values)
    y_max = np.percentile(values, ylim_percentile)
    margin = (y_max - y_min) * 0.05  # Add 5% margin
    ax.set_ylim(y_min - margin, y_max + margin)

  # Create custom legend
  from matplotlib.lines import Line2D

  legend_elements = [
    Line2D([0], [0], color="red", linewidth=2, label="Median"),
    Line2D(
      [0],
      [0],
      marker="D",
      color="w",
      markerfacecolor="green",
      markersize=8,
      label="Mean",
    ),
  ]
  if show_outliers:
    legend_elements.append(
      Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=5,
        label="Outliers",
      )
    )
  ax.legend(handles=legend_elements, fontsize=10, loc="best")

  if show_stats:
    stats_text = (
      f"n = {len(values)}\n"
      f"Mean: {mean_val:.4f}\n"
      f"Std: {std_val:.4f}\n"
      f"Median: {median_val:.4f}\n"
      f"Q1: {q1:.4f}\n"
      f"Q3: {q3:.4f}\n"
      f"IQR: {iqr:.4f}\n"
      f"Min: {min_val:.4f}\n"
      f"Max: {max_val:.4f}\n"
      f"Outliers: {n_outliers} ({outlier_percentage:.1f}%)"
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


def plot_metric_boxplot_comparison(
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
  figsize: tuple[int, int] = (12, 6),
  colors: list[str] | None = None,
  show_outliers: bool = True,
  remove_outliers: bool = False,
  show_legend: bool = True,
  show_stats_summary: bool = True,
  ylim_percentile: float | None = None,
) -> None:
  client = SQLiteClient(db_filename)

  if colors is None:
    colors = plt.cm.Set2(np.linspace(0, 1, len(group_values)))
    colors = [plt.cm.colors.rgb2hex(c) for c in colors]

  fig, ax = plt.subplots(figsize=figsize)

  all_data = []
  all_stats = []
  positions = []
  valid_group_values = []  # Track which groups actually have data

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

    # Store original count for reporting
    original_count = len(values)

    # Remove outliers if requested (before calculating statistics)
    if remove_outliers:
      q1_temp = np.percentile(values, 25)
      q3_temp = np.percentile(values, 75)
      iqr_temp = q3_temp - q1_temp
      lower_bound_temp = q1_temp - 1.5 * iqr_temp
      upper_bound_temp = q3_temp + 1.5 * iqr_temp
      outlier_mask = (values >= lower_bound_temp) & (values <= upper_bound_temp)
      values = values[outlier_mask]
      removed_count = original_count - len(values)
      print(
        f"{group_value}: Removed {removed_count} outliers "
        f"({removed_count / original_count * 100:.2f}%)"
      )

    all_data.append(values)
    positions.append(len(all_data))  # Use sequential positions starting from 1
    valid_group_values.append(group_value)  # Track this group has data

    mean_val = np.mean(values)
    std_val = np.std(values)
    median_val = np.median(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    # Count outliers for this group (from the current data, after potential removal)
    n_outliers, _ = count_outliers(values)

    all_stats.append(
      {
        "group": group_value,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "q1": q1,
        "q3": q3,
        "n": len(values),
        "outliers": n_outliers,
      }
    )

  if not all_data:
    print("No data found for any of the specified groups")
    return

  box_parts = ax.boxplot(
    all_data,
    positions=positions,
    vert=True,
    patch_artist=True,
    showfliers=show_outliers,
    widths=0.6,
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(
      marker="o",
      markerfacecolor="red",
      markersize=4,
      alpha=0.4,
      linestyle="none",
    ),
  )

  # Color each box
  for patch, color in zip(box_parts["boxes"], colors[: len(all_data)], strict=False):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    patch.set_linewidth(1.5)

  # Add mean markers
  for i, (pos, data) in enumerate(zip(positions, all_data, strict=False)):
    mean_val = np.mean(data)
    ax.plot(pos, mean_val, marker="D", color="darkgreen", markersize=8, zorder=3)

  metric_labels = {
    "prd": "PRD (%)",
    "nmse": "NMSE",
    "sndr": "SNDR (dB)",
    "ssim": "SSIM",
    "elapsed_time_s": "Elapsed Time (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)
  ax.set_xlabel(group_by.replace("_", " ").title(), fontsize=12)
  ax.set_xticks(positions)
  ax.set_xticklabels(valid_group_values, rotation=45, ha="right")

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
    title = f"Boxplot Comparison of {metric_labels.get(metric, metric.upper())} by {group_by.replace('_', ' ').title()}"
    if filter_str:
      title += f"\n{filter_str}"

  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, alpha=0.3, axis="y")

  # Set y-axis limits to focus on main distribution if specified
  if ylim_percentile is not None and show_outliers and all_data:
    all_values = np.concatenate(all_data)
    y_min = np.min(all_values)
    y_max = np.percentile(all_values, ylim_percentile)
    margin = (y_max - y_min) * 0.05  # Add 5% margin
    ax.set_ylim(y_min - margin, y_max + margin)

  if show_legend:
    from matplotlib.lines import Line2D

    legend_elements = [
      Line2D([0], [0], color="red", linewidth=2, label="Median"),
      Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="darkgreen",
        markersize=8,
        label="Mean",
      ),
    ]
    if show_outliers:
      legend_elements.append(
        Line2D(
          [0],
          [0],
          marker="o",
          color="w",
          markerfacecolor="red",
          markersize=4,
          label="Outliers",
        )
      )
    ax.legend(handles=legend_elements, fontsize=10, loc="best")

  plt.tight_layout()
  plt.show()

  if show_stats_summary:
    print(f"\n{'=' * 105}")
    print(f"Statistics Summary by {group_by}:")
    print(f"{'=' * 105}")
    print(
      f"{'Group':20s} | {'n':>5s} | {'Mean':>10s} | {'Std':>10s} | "
      f"{'Median':>10s} | {'Q1':>10s} | {'Q3':>10s} | {'Outliers':>10s}"
    )
    print(f"{'-' * 105}")
    for stat in all_stats:
      outlier_pct = (stat["outliers"] / stat["n"]) * 100
      print(
        f"{stat['group']:20s} | {stat['n']:5d} | "
        f"{stat['mean']:10.4f} | {stat['std']:10.4f} | "
        f"{stat['median']:10.4f} | {stat['q1']:10.4f} | {stat['q3']:10.4f} | "
        f"{stat['outliers']:6d} ({outlier_pct:4.1f}%)"
      )
    print(f"{'=' * 105}\n")


def main() -> None:
  # Example 1: Single boxplot with filters
  plot_metric_boxplot(
    db_filename="eeg_cs_evaluations.db",
    dataset="chbmit",
    metric="nmse",
    sensing_matrix="BPBD",
    sparsifying_matrix="Wavelet_db4_5L",
    algorithm="CVXP_BP_IND_CLARABEL",
    compression_rate=2,
    show_outliers=False,  # This removes the outliers
  )

  # # Example 2: Comparison boxplot
  # plot_metric_boxplot_comparison(
  #   db_filename="db.sqlite3",
  #   metric="nmse",
  #   group_by="sparsifying_matrix",
  #   group_values=["DCT", "IDCT"],
  #   dataset="chbmit",
  #   sensing_matrix="BPBD",
  #   algorithm="BPDN",
  #   compression_rate=2,
  #   show_outliers=False,  # Don't show outlier points on plot
  #   remove_outliers=False,  # Remove outliers from data (affects statistics)
  #   # Use linear scale and clip to 99.5th percentile to see boxes clearly
  #   # ylim_percentile=99,
  # )


if __name__ == "__main__":
  main()
