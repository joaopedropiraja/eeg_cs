from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from eeg_cs.db.client import SQLiteClient


def plot_metric_line(
  db_filename: str,
  metric: str,
  x_axis: str,
  sensing_matrix: str | None = None,
  sparsifying_matrix: str | None = None,
  algorithm: str | None = None,
  dataset: str | None = None,
  channel: str | None = None,
  compression_rate: int | None = None,
  segment_length_s: int | None = None,
  title: str | None = None,
  figsize: tuple[int, int] = (10, 6),
  color: str = "steelblue",
  marker: str = "o",
  show_stats_table: bool = True,
) -> None:
  """Plot a line graph showing mean ± std for a metric across x_axis values.

  Args:
      db_filename: Path to the SQLite database file
      metric: Metric to plot (prd, nmse, sndr, ssim, elapsed_time_s, mean_freq)
      x_axis: Parameter to vary on x-axis (e.g., 'compression_rate', 'segment_length_s')
      sensing_matrix: Filter by sensing matrix name
      sparsifying_matrix: Filter by sparsifying matrix name
      algorithm: Filter by algorithm name
      dataset: Filter by dataset name
      channel: Filter by channel name
      compression_rate: Filter by compression rate (ignored if x_axis='compression_rate')
      segment_length_s: Filter by segment length (ignored if x_axis='segment_length_s')
      title: Custom title for the plot
      figsize: Figure size as (width, height)
      color: Line and marker color
      marker: Marker style
      show_stats_table: Whether to print statistics table
  """
  client = SQLiteClient(db_filename)

  # Build base query
  query = f"SELECT {x_axis}, {metric} FROM evaluations WHERE 1=1"
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

  # Don't filter by the x_axis parameter
  if compression_rate is not None and x_axis != "compression_rate":
    query += " AND compression_rate = ?"
    params.append(compression_rate)

  if segment_length_s is not None and x_axis != "segment_length_s":
    query += " AND segment_length_s = ?"
    params.append(segment_length_s)

  query += f" ORDER BY {x_axis}"

  results = client.query_all(query, params)

  if not results:
    print("No data found for the specified filters")
    return

  # Group data by x_axis values
  data_by_x: dict[float, list[float]] = {}
  for row in results:
    x_val = row[x_axis]
    metric_val = row[metric]
    if x_val not in data_by_x:
      data_by_x[x_val] = []
    data_by_x[x_val].append(metric_val)

  # Calculate mean and std for each x value
  x_values = sorted(data_by_x.keys())
  means = []
  stds = []
  counts = []

  for x_val in x_values:
    values = np.array(data_by_x[x_val])
    means.append(np.mean(values))
    stds.append(np.std(values))
    counts.append(len(values))

  means = np.array(means)
  stds = np.array(stds)
  x_values = np.array(x_values)

  # Create plot
  fig, ax = plt.subplots(figsize=figsize)

  # Plot line with error bars
  ax.errorbar(
    x_values,
    means,
    yerr=stds,
    fmt=marker,
    color=color,
    ecolor=color,
    elinewidth=2,
    capsize=5,
    capthick=2,
    markersize=8,
    linewidth=2,
    alpha=0.8,
  )

  # Labels
  metric_labels = {
    "prd": "PRD (%)",
    "nmse": "NMSE",
    "sndr": "SNDR (dB)",
    "ssim": "SSIM",
    "elapsed_time_s": "Elapsed Time (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  x_axis_labels = {
    "compression_rate": "Compression Rate",
    "segment_length_s": "Segment Length (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  ax.set_xlabel(
    x_axis_labels.get(x_axis, x_axis.replace("_", " ").title()), fontsize=12
  )
  ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)

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
    if compression_rate and x_axis != "compression_rate":
      filter_parts.append(f"CR: {compression_rate}")
    if segment_length_s and x_axis != "segment_length_s":
      filter_parts.append(f"Segment: {segment_length_s}s")

    filter_str = " | ".join(filter_parts) if filter_parts else "All Data"
    title = f"{metric_labels.get(metric, metric.upper())} vs {x_axis_labels.get(x_axis, x_axis)}\n{filter_str}"

  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()

  # Print statistics table
  if show_stats_table:
    print(f"\n{'=' * 80}")
    print(
      f"Statistics: {metric_labels.get(metric, metric.upper())} by {x_axis_labels.get(x_axis, x_axis)}"
    )
    print(f"{'=' * 80}")
    print(
      f"{x_axis_labels.get(x_axis, x_axis):20s} | {'n':>6s} | {'Mean':>10s} | {'Std':>10s}"
    )
    print(f"{'-' * 80}")
    for x_val, mean_val, std_val, count in zip(
      x_values, means, stds, counts, strict=False
    ):
      print(f"{x_val:20.2f} | {count:6d} | {mean_val:10.4f} | {std_val:10.4f}")
    print(f"{'=' * 80}\n")


def plot_metric_line_comparison(
  db_filename: str,
  metric: str,
  x_axis: str,
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
  markers: list[str] | None = None,
  show_legend: bool = True,
  show_stats_table: bool = True,
) -> None:
  """Plot multiple line graphs comparing different groups.

  Args:
      db_filename: Path to the SQLite database file
      metric: Metric to plot (prd, nmse, sndr, ssim, elapsed_time_s, mean_freq)
      x_axis: Parameter to vary on x-axis (e.g., 'compression_rate', 'segment_length_s')
      group_by: Column to group by (e.g., 'sensing_matrix', 'algorithm')
      group_values: List of values to compare for the group_by column
      sensing_matrix: Filter by sensing matrix name
      sparsifying_matrix: Filter by sparsifying matrix name
      algorithm: Filter by algorithm name
      dataset: Filter by dataset name
      channel: Filter by channel name
      compression_rate: Filter by compression rate (ignored if x_axis='compression_rate')
      segment_length_s: Filter by segment length (ignored if x_axis='segment_length_s')
      title: Custom title for the plot
      figsize: Figure size as (width, height)
      colors: List of colors for each group
      markers: List of marker styles for each group
      show_legend: Whether to show legend
      show_stats_table: Whether to print statistics table
  """
  client = SQLiteClient(db_filename)

  if colors is None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

  if markers is None:
    markers = ["o", "s", "^", "D", "v", "p"]

  fig, ax = plt.subplots(figsize=figsize)

  all_stats = []

  for i, group_value in enumerate(group_values):
    # Build query
    query = f"SELECT {x_axis}, {metric} FROM evaluations WHERE {group_by} = ?"
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

    # Don't filter by the x_axis parameter or group_by parameter
    if (
      compression_rate is not None
      and x_axis != "compression_rate"
      and group_by != "compression_rate"
    ):
      query += " AND compression_rate = ?"
      params.append(compression_rate)

    if (
      segment_length_s is not None
      and x_axis != "segment_length_s"
      and group_by != "segment_length_s"
    ):
      query += " AND segment_length_s = ?"
      params.append(segment_length_s)

    query += f" ORDER BY {x_axis}"

    results = client.query_all(query, params)

    if not results:
      print(f"No data found for {group_by} = {group_value}")
      continue

    # Group data by x_axis values
    data_by_x: dict[float, list[float]] = {}
    for row in results:
      x_val = row[x_axis]
      metric_val = row[metric]
      if x_val not in data_by_x:
        data_by_x[x_val] = []
      data_by_x[x_val].append(metric_val)

    # Calculate mean and std for each x value
    x_values = sorted(data_by_x.keys())
    means = []
    stds = []
    counts = []

    for x_val in x_values:
      values = np.array(data_by_x[x_val])
      means.append(np.mean(values))
      stds.append(np.std(values))
      counts.append(len(values))

    means = np.array(means)
    stds = np.array(stds)
    x_values = np.array(x_values)

    # Plot line with error bars
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    ax.errorbar(
      x_values,
      means,
      yerr=stds,
      fmt=marker,
      color=color,
      ecolor=color,
      elinewidth=2,
      capsize=5,
      capthick=2,
      markersize=8,
      linewidth=2,
      alpha=0.8,
      label=group_value,
    )

    # Store stats for table
    for x_val, mean_val, std_val, count in zip(
      x_values, means, stds, counts, strict=False
    ):
      all_stats.append(
        {
          "group": group_value,
          "x_value": x_val,
          "mean": mean_val,
          "std": std_val,
          "n": count,
        }
      )

  # Labels
  metric_labels = {
    "prd": "PRD (%)",
    "nmse": "NMSE",
    "sndr": "SNDR (dB)",
    "ssim": "SSIM",
    "elapsed_time_s": "Elapsed Time (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  x_axis_labels = {
    "compression_rate": "Compression Rate",
    "segment_length_s": "Segment Length (s)",
    "mean_freq": "Mean Frequency (Hz)",
  }

  ax.set_xlabel(
    x_axis_labels.get(x_axis, x_axis.replace("_", " ").title()), fontsize=12
  )
  ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)

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
    if (
      compression_rate
      and x_axis != "compression_rate"
      and group_by != "compression_rate"
    ):
      filter_parts.append(f"CR: {compression_rate}")
    if (
      segment_length_s
      and x_axis != "segment_length_s"
      and group_by != "segment_length_s"
    ):
      filter_parts.append(f"Segment: {segment_length_s}s")

    filter_str = " | ".join(filter_parts) if filter_parts else ""
    title = f"{metric_labels.get(metric, metric.upper())} vs {x_axis_labels.get(x_axis, x_axis)} by {group_by.replace('_', ' ').title()}"
    if filter_str:
      title += f"\n{filter_str}"

  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, alpha=0.3)

  if show_legend:
    ax.legend(fontsize=10, loc="best")

  plt.tight_layout()
  plt.show()

  # Print statistics table
  if show_stats_table:
    print(f"\n{'=' * 90}")
    print(
      f"Statistics: {metric_labels.get(metric, metric.upper())} by {x_axis_labels.get(x_axis, x_axis)} and {group_by}"
    )
    print(f"{'=' * 90}")
    print(
      f"{'Group':20s} | {x_axis_labels.get(x_axis, x_axis):>15s} | {'n':>6s} | {'Mean':>10s} | {'Std':>10s}"
    )
    print(f"{'-' * 90}")
    for stat in all_stats:
      print(
        f"{stat['group']:20s} | {stat['x_value']:15.2f} | {stat['n']:6d} | "
        f"{stat['mean']:10.4f} | {stat['std']:10.4f}"
      )
    print(f"{'=' * 90}\n")


def main() -> None:
  # plot_metric_line(
  #   db_filename="db.sqlite3",
  #   metric="nmse",
  #   x_axis="sensing_matrix",
  #   sparsifying_matrix="DCT",
  #   algorithm="BPDN",
  # )

  # Example 2: Comparison line plot - SNDR vs Compression Rate for different algorithms
  # plot_metric_line_comparison(
  #   db_filename="db.sqlite3",
  #   metric="nmse",
  #   x_axis="sensing_matrix",
  #   group_by="sparsifying_matrix",
  #   group_values=["DCT", "IDCT"],
  #   sensing_matrix="BPBD",
  #   algorithm="BPDN",
  #   dataset="chbmit",
  # )

  db_filename = "eeg_cs_evaluations.db"
  metric = "nmse"

  x_axis = "sensing_matrix"
  pattern = "SB_[0-9]*"
  transform_x_axis = lambda x: int(x.split("_")[1])

  # x_axis = "algorithm"
  # pattern = "OMP_k[0-9]*"
  # transform_x_axis = lambda x: int(re.fullmatch(r"OMP_k(\d+)", x).group(1))

  x_axis_label = "Block Size"
  y_axis_label = "NMSE"
  figsize = (10, 6)
  use_confidence_interval = True  # Set to False to use standard deviation
  confidence_level = 0.95  # 95% confidence interval

  client = SQLiteClient(db_filename)

  # Build base query
  query = f"SELECT {metric}, {x_axis} FROM evaluations WHERE {x_axis} GLOB ?"
  params: list[str | int] = [pattern]

  results = client.query_all(query, params)

  if not results:
    print("No data found for the specified filters")
    return

  # Group data by x_axis values
  data_by_x: dict[float, list[float]] = {}
  for row in results:
    x_val = transform_x_axis(row[x_axis])
    metric_val = row[metric]
    if x_val not in data_by_x:
      data_by_x[x_val] = []
    data_by_x[x_val].append(metric_val)

  # Calculate mean and error bars for each x value
  x_values = sorted(data_by_x.keys())
  means = []
  errors = []  # Will be either std or confidence interval
  counts = []

  for x_val in x_values:
    values = np.array(data_by_x[x_val])
    mean = np.mean(values)
    n = len(values)
    means.append(mean)
    counts.append(n)

    if use_confidence_interval:
      # Calculate confidence interval
      std_error = np.std(values, ddof=1) / np.sqrt(n)  # Standard error
      # Critical value for confidence interval (1.96 for 95%, 2.576 for 99%)
      if confidence_level == 0.95:
        critical_value = 1.96
      elif confidence_level == 0.99:
        critical_value = 2.576
      elif confidence_level == 0.90:
        critical_value = 1.645
      else:
        # General case using scipy (falls back to 1.96 if not available)
        try:
          from scipy import stats

          critical_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        except ImportError:
          critical_value = 1.96
      error = critical_value * std_error
    else:
      # Use standard deviation
      error = np.std(values)

    errors.append(error)

  means = np.array(means)
  errors = np.array(errors)
  x_values = np.array(x_values)

  # Create plot
  fig, ax = plt.subplots(figsize=figsize)

  # Plot line with error bars
  ax.errorbar(
    x_values,
    means,
    yerr=errors,
    marker="o",
    elinewidth=2,
    capsize=5,
    capthick=2,
    markersize=8,
    linewidth=2,
    alpha=0.8,
  )

  ax.set_xlabel(x_axis_label, fontsize=12)
  ax.set_ylabel(y_axis_label, fontsize=12)

  # Set y-axis limits based on error bars
  y_min = min(means - errors)
  y_max = max(means + errors)
  margin = (y_max - y_min) * 0.1  # Add 10% margin
  ax.set_ylim(y_min - margin, y_max + margin)

  # Set x-axis ticks to show all values
  ax.set_xticks(x_values)
  ax.set_xticklabels([int(x) for x in x_values])

  # Add title indicating error bar type
  error_type = (
    f"{int(confidence_level * 100)}% CI" if use_confidence_interval else "Std Dev"
  )
  ax.set_title(
    f"{y_axis_label} vs {x_axis_label} (Error bars: {error_type})", fontsize=12
  )

  ax.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()
