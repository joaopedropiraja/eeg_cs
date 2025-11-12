"""Identify and analyze outliers for specific metrics from the evaluations database.

This module provides functionality to:
- Identify outliers using the IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- Export outlier records to CSV for analysis
- Display statistics about outliers vs non-outliers
- Filter by sensing matrix, sparsifying matrix, algorithm, dataset, channel, etc.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from eeg_cs.db.client import SQLiteClient


def identify_outliers(
  db_filename: str,
  metric: str,
  sensing_matrix: str | None = None,
  sparsifying_matrix: str | None = None,
  algorithm: str | None = None,
  dataset: str | None = None,
  channel: int | None = None,
  compression_rate: float | None = None,
  segment_length_s: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
  """Identify outliers for a specific metric with optional filters.

  Args:
      metric: Metric name ('prd', 'nmse', 'sndr', 'ssim', 'elapsed_time')
      sensing_matrix: Optional filter for sensing matrix
      sparsifying_matrix: Optional filter for sparsifying matrix
      algorithm: Optional filter for algorithm
      dataset: Optional filter for dataset
      channel: Optional filter for channel
      compression_rate: Optional filter for compression rate
      segment_length_s: Optional filter for segment length in seconds

  Returns:
      Tuple of (outliers_df, non_outliers_df, statistics_dict):
      - outliers_df: DataFrame with outlier records
      - non_outliers_df: DataFrame with non-outlier records
      - statistics_dict: Dictionary with Q1, Q3, IQR, lower/upper bounds, counts, etc.
  """
  client = SQLiteClient(db_filename)

  # Build filter conditions
  conditions = []
  if sensing_matrix:
    conditions.append(f"sensing_matrix = '{sensing_matrix}'")
  if sparsifying_matrix:
    conditions.append(f"sparsifying_matrix = '{sparsifying_matrix}'")
  if algorithm:
    conditions.append(f"algorithm = '{algorithm}'")
  if dataset:
    conditions.append(f"dataset = '{dataset}'")
  if channel is not None:
    conditions.append(f"channel = {channel}")
  if compression_rate is not None:
    conditions.append(f"compression_rate = {compression_rate}")
  if segment_length_s is not None:
    conditions.append(f"segment_length_s = {segment_length_s}")

  where_clause = " AND ".join(conditions) if conditions else "1=1"

  # Query all data
  query = f"""
        SELECT 
            id, sensing_matrix, sparsifying_matrix, algorithm, dataset, 
            channel, compression_rate, segment_length_s, start_time_idx,
            prd, nmse, sndr, ssim, elapsed_time_s, mean_freq, fs, file_name
        FROM evaluations
        WHERE {where_clause}
        ORDER BY {metric} DESC
    """

  results = client.query_all(query)

  if not results:
    return pd.DataFrame(), pd.DataFrame(), {}

  # Convert to DataFrame (results are already dictionaries)
  df = pd.DataFrame(results)

  # Rename elapsed_time_s to elapsed_time for consistency
  df = df.rename(
    columns={"elapsed_time_s": "elapsed_time", "start_time_idx": "segment_idx"}
  )

  # Calculate IQR statistics
  values = df[metric].values
  q1 = np.percentile(values, 25)
  q3 = np.percentile(values, 75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr

  # Identify outliers
  outlier_mask = (values < lower_bound) | (values > upper_bound)
  outliers_df = df[outlier_mask].copy()
  non_outliers_df = df[~outlier_mask].copy()

  # Calculate statistics
  stats = {
    "metric": metric,
    "total_records": len(df),
    "n_outliers": len(outliers_df),
    "n_non_outliers": len(non_outliers_df),
    "outlier_percentage": (len(outliers_df) / len(df)) * 100,
    "q1": q1,
    "q3": q3,
    "iqr": iqr,
    "lower_bound": lower_bound,
    "upper_bound": upper_bound,
    "mean_all": np.mean(values),
    "std_all": np.std(values),
    "median_all": np.median(values),
    "min_all": np.min(values),
    "max_all": np.max(values),
  }

  if len(outliers_df) > 0:
    stats["mean_outliers"] = np.mean(outliers_df[metric].values)
    stats["std_outliers"] = np.std(outliers_df[metric].values)
    stats["median_outliers"] = np.median(outliers_df[metric].values)
    stats["min_outliers"] = np.min(outliers_df[metric].values)
    stats["max_outliers"] = np.max(outliers_df[metric].values)

  if len(non_outliers_df) > 0:
    stats["mean_non_outliers"] = np.mean(non_outliers_df[metric].values)
    stats["std_non_outliers"] = np.std(non_outliers_df[metric].values)
    stats["median_non_outliers"] = np.median(non_outliers_df[metric].values)
    stats["min_non_outliers"] = np.min(non_outliers_df[metric].values)
    stats["max_non_outliers"] = np.max(non_outliers_df[metric].values)

  return outliers_df, non_outliers_df, stats


def export_outliers_to_csv(
  outliers_df: pd.DataFrame,
  metric: str,
  output_dir: str = "outliers_analysis",
  filename_suffix: str = "",
) -> Path:
  """Export outliers DataFrame to CSV file.

  Args:
      outliers_df: DataFrame containing outlier records
      metric: Name of the metric being analyzed
      output_dir: Directory to save the CSV file
      filename_suffix: Optional suffix to add to filename

  Returns:
      Path to the created CSV file
  """
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  suffix = f"_{filename_suffix}" if filename_suffix else ""
  filename = f"outliers_{metric}{suffix}.csv"
  filepath = output_path / filename

  outliers_df.to_csv(filepath, index=False)
  return filepath


def print_outlier_statistics(stats: dict[str, float]) -> None:
  """Print formatted outlier statistics.

  Args:
      stats: Dictionary containing statistics from identify_outliers()
  """
  print("=" * 80)
  print(f"Outlier Analysis for Metric: {stats['metric'].upper()}")
  print("=" * 80)
  print("\nData Summary:")
  print(f"  Total records:        {stats['total_records']:,}")
  print(
    f"  Outliers:             {stats['n_outliers']:,} ({stats['outlier_percentage']:.2f}%)"
  )
  print(f"  Non-outliers:         {stats['n_non_outliers']:,}")

  print("\nIQR Bounds:")
  print(f"  Q1:                   {stats['q1']:.6f}")
  print(f"  Q3:                   {stats['q3']:.6f}")
  print(f"  IQR:                  {stats['iqr']:.6f}")
  print(f"  Lower bound:          {stats['lower_bound']:.6f}")
  print(f"  Upper bound:          {stats['upper_bound']:.6f}")

  print("\nAll Data Statistics:")
  print(f"  Mean ± Std:           {stats['mean_all']:.6f} ± {stats['std_all']:.6f}")
  print(f"  Median:               {stats['median_all']:.6f}")
  print(f"  Range:                [{stats['min_all']:.6f}, {stats['max_all']:.6f}]")

  if stats.get("mean_outliers") is not None:
    print("\nOutliers Statistics:")
    print(
      f"  Mean ± Std:           {stats['mean_outliers']:.6f} ± {stats['std_outliers']:.6f}"
    )
    print(f"  Median:               {stats['median_outliers']:.6f}")
    print(
      f"  Range:                [{stats['min_outliers']:.6f}, {stats['max_outliers']:.6f}]"
    )

  if stats.get("mean_non_outliers") is not None:
    print("\nNon-Outliers Statistics:")
    print(
      f"  Mean ± Std:           {stats['mean_non_outliers']:.6f} ± {stats['std_non_outliers']:.6f}"
    )
    print(f"  Median:               {stats['median_non_outliers']:.6f}")
    print(
      f"  Range:                [{stats['min_non_outliers']:.6f}, {stats['max_non_outliers']:.6f}]"
    )

  print("=" * 80)


def analyze_outlier_patterns(outliers_df: pd.DataFrame, metric: str) -> None:
  """Analyze and print patterns in outlier data.

  Args:
      outliers_df: DataFrame containing outlier records
      metric: Name of the metric being analyzed
  """
  if len(outliers_df) == 0:
    print("\nNo outliers found!")
    return

  print("\n" + "=" * 80)
  print("Outlier Patterns Analysis")
  print("=" * 80)

  # Count by sensing matrix
  print("\nOutliers by Sensing Matrix:")
  sensing_counts = outliers_df["sensing_matrix"].value_counts()
  for matrix, count in sensing_counts.items():
    pct = (count / len(outliers_df)) * 100
    print(f"  {matrix:20s}: {count:4d} ({pct:5.2f}%)")

  # Count by sparsifying matrix
  print("\nOutliers by Sparsifying Matrix:")
  sparsifying_counts = outliers_df["sparsifying_matrix"].value_counts()
  for matrix, count in sparsifying_counts.items():
    pct = (count / len(outliers_df)) * 100
    print(f"  {matrix:20s}: {count:4d} ({pct:5.2f}%)")

  # Count by algorithm
  print("\nOutliers by Algorithm:")
  algo_counts = outliers_df["algorithm"].value_counts()
  for algo, count in algo_counts.items():
    pct = (count / len(outliers_df)) * 100
    print(f"  {algo:20s}: {count:4d} ({pct:5.2f}%)")

  # Count by compression rate
  print("\nOutliers by Compression Rate:")
  cr_counts = outliers_df["compression_rate"].value_counts().sort_index()
  for cr, count in cr_counts.items():
    pct = (count / len(outliers_df)) * 100
    print(f"  {cr:.2f}:               {count:4d} ({pct:5.2f}%)")

  # Top 10 worst outliers
  print(f"\nTop 10 Worst Outliers (by {metric}):")
  top_outliers = outliers_df.nlargest(10, metric)
  print(
    f"{'ID':>6} | {'Sensing':15s} | {'Sparsifying':15s} | {'Algorithm':10s} | {'CR':>5s} | {metric.upper():>10s}"
  )
  print("-" * 80)
  for _, row in top_outliers.iterrows():
    print(
      f"{row['id']:6d} | {row['sensing_matrix']:15s} | {row['sparsifying_matrix']:15s} | "
      f"{row['algorithm']:10s} | {row['compression_rate']:5.2f} | {row[metric]:10.6f}"
    )

  print("=" * 80)


def analyze_outlier_properties(
  outliers_df: pd.DataFrame, non_outliers_df: pd.DataFrame, metric: str
) -> None:
  """Analyze specific properties of outliers like mean frequency, channels, etc.

  Args:
      outliers_df: DataFrame containing outlier records
      non_outliers_df: DataFrame containing non-outlier records
      metric: Name of the metric being analyzed
  """
  if len(outliers_df) == 0:
    print("\nNo outliers found!")
    return

  print("\n" + "=" * 80)
  print("Outlier Properties Analysis")
  print("=" * 80)

  # Mean Frequency Analysis
  if "mean_freq" in outliers_df.columns:
    print("\nMean Frequency Analysis:")
    outlier_mean_freq = outliers_df["mean_freq"].values
    non_outlier_mean_freq = non_outliers_df["mean_freq"].values

    print("  Outliers:")
    print(
      f"    Mean ± Std:         {np.mean(outlier_mean_freq):.4f} ± {np.std(outlier_mean_freq):.4f} Hz"
    )
    print(f"    Median:             {np.median(outlier_mean_freq):.4f} Hz")
    print(
      f"    Range:              [{np.min(outlier_mean_freq):.4f}, {np.max(outlier_mean_freq):.4f}] Hz"
    )

    print("\n  Non-Outliers:")
    print(
      f"    Mean ± Std:         {np.mean(non_outlier_mean_freq):.4f} ± {np.std(non_outlier_mean_freq):.4f} Hz"
    )
    print(f"    Median:             {np.median(non_outlier_mean_freq):.4f} Hz")
    print(
      f"    Range:              [{np.min(non_outlier_mean_freq):.4f}, {np.max(non_outlier_mean_freq):.4f}] Hz"
    )

    # Statistical comparison
    mean_diff = np.mean(outlier_mean_freq) - np.mean(non_outlier_mean_freq)
    print("\n  Difference (Outliers - Non-Outliers):")
    print(f"    Mean difference:    {mean_diff:+.4f} Hz")

  # Channel Analysis
  if "channel" in outliers_df.columns:
    print("\n" + "-" * 80)
    print("Channel Analysis:")

    # Outliers by channel
    print("\n  Outliers by Channel:")
    channel_counts = outliers_df["channel"].value_counts().sort_index()
    total_outliers = len(outliers_df)

    for channel, count in channel_counts.items():
      pct = (count / total_outliers) * 100
      print(f"    Channel {channel:3s}: {count:5d} outliers ({pct:5.2f}%)")

    # Calculate outlier rate per channel
    print("\n  Outlier Rate by Channel:")
    all_channels = pd.concat([outliers_df["channel"], non_outliers_df["channel"]])
    channel_totals = all_channels.value_counts().sort_index()

    outlier_rates = []
    for channel in channel_totals.index:
      outlier_count = channel_counts.get(channel, 0)
      total_count = channel_totals[channel]
      outlier_rate = (outlier_count / total_count) * 100
      outlier_rates.append((channel, outlier_count, total_count, outlier_rate))

    # Sort by outlier rate descending
    outlier_rates.sort(key=lambda x: x[3], reverse=True)

    print(f"    {'Channel':8s} | {'Outliers':>8s} | {'Total':>8s} | {'Rate':>8s}")
    print("    " + "-" * 42)
    for channel, out_count, tot_count, rate in outlier_rates[:15]:  # Show top 15
      print(f"    {channel:8s} | {out_count:8d} | {tot_count:8d} | {rate:7.2f}%")

  # Dataset Analysis
  if "dataset" in outliers_df.columns:
    print("\n" + "-" * 80)
    print("Dataset Analysis:")

    dataset_counts = outliers_df["dataset"].value_counts()
    for dataset, count in dataset_counts.items():
      pct = (count / len(outliers_df)) * 100
      print(f"  {dataset:20s}: {count:5d} outliers ({pct:5.2f}%)")

  # File Name Analysis (top files with most outliers)
  if "file_name" in outliers_df.columns:
    print("\n" + "-" * 80)
    print("Top 10 Files with Most Outliers:")

    file_counts = outliers_df["file_name"].value_counts().head(10)
    for file_name, count in file_counts.items():
      pct = (count / len(outliers_df)) * 100
      print(f"  {file_name:30s}: {count:4d} ({pct:5.2f}%)")

  # Sampling Frequency Analysis
  if "fs" in outliers_df.columns:
    print("\n" + "-" * 80)
    print("Sampling Frequency Analysis:")

    fs_counts = outliers_df["fs"].value_counts().sort_index()
    for fs, count in fs_counts.items():
      pct = (count / len(outliers_df)) * 100
      print(f"  {fs:5d} Hz: {count:5d} outliers ({pct:5.2f}%)")

  print("=" * 80)


def main() -> None:
  """Main function demonstrating outlier analysis."""
  # Example: Analyze NMSE outliers
  metric = "nmse"

  print(f"Analyzing outliers for metric: {metric}")
  print("Filters: sparsifying_matrix='DCT'\n")

  outliers_df, non_outliers_df, stats = identify_outliers(
    db_filename="eeg_cs_evaluations.db",
    metric=metric,
    compression_rate=8,
    dataset="chbmit",
    sensing_matrix="BPBD",
    sparsifying_matrix="Wavelet_db4_5L",
    algorithm="CVXP_BP_IND_CLARABEL",
  )
  print(stats)
  # Print statistics
  print_outlier_statistics(stats)

  # Analyze patterns
  analyze_outlier_patterns(outliers_df, metric)

  # Analyze specific properties
  analyze_outlier_properties(outliers_df, non_outliers_df, metric)

  # Export to CSV
  if len(outliers_df) > 0:
    filepath = export_outliers_to_csv(
      outliers_df, metric, output_dir="outliers_analysis", filename_suffix="dct"
    )
    print(f"\nOutliers exported to: {filepath}")
    print(f"Total records exported: {len(outliers_df)}")

  # Show sample of outliers
  if len(outliers_df) > 0:
    print("\nSample of outlier records (first 5):")
    print(
      outliers_df[
        ["id", "sensing_matrix", "algorithm", "compression_rate", metric]
      ].head()
    )


if __name__ == "__main__":
  main()
