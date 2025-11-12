"""Build statistics table from evaluation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eeg_cs.db.client import SQLiteClient


def build_statistics_table(db_filename: str = "eeg_cs/db/db.sqlite3") -> pd.DataFrame:
  """
  Build a statistics table from the database.

  Groups results by sensing_matrix, sparsifying_matrix, and algorithm,
  then calculates mean and standard deviation for nmse, prd, sndr, and elapsed_time_s.

  Args:
      db_filename: Path to the database file (relative to project root)

  Returns:
      DataFrame with statistics
  """

  client = SQLiteClient(db_filename)

  # Query to get aggregated statistics
  sql = """
    SELECT 
        sensing_matrix,
        sparsifying_matrix,
        algorithm,
        AVG(nmse) as avg_nmse,
        AVG(prd) as avg_prd,
        AVG(sndr) as avg_sndr,
        AVG(elapsed_time_s) as avg_elapsed_time,
        COUNT(*) as n_samples
    FROM evaluations
    GROUP BY sensing_matrix, sparsifying_matrix, algorithm
    ORDER BY sensing_matrix, sparsifying_matrix, algorithm
    """

  results = client.query_all(sql)

  if not results:
    print("No data found in the database.")
    return pd.DataFrame()

  # Get individual records to calculate standard deviations
  sql_detail = """
    SELECT 
        sensing_matrix,
        sparsifying_matrix,
        algorithm,
        nmse,
        prd,
        sndr,
        elapsed_time_s
    FROM evaluations
    ORDER BY sensing_matrix, sparsifying_matrix, algorithm
    """

  detail_results = client.query_all(sql_detail)
  df_detail = pd.DataFrame(detail_results)

  # Calculate statistics grouped by sensing_matrix, sparsifying_matrix, and algorithm
  grouped = df_detail.groupby(["sensing_matrix", "sparsifying_matrix", "algorithm"])

  stats = grouped.agg(
    {
      "nmse": ["mean", "std"],
      "prd": ["mean", "std"],
      "sndr": ["mean", "std"],
      "elapsed_time_s": ["mean", "std"],
    }
  ).reset_index()

  # Flatten column names
  stats.columns = [
    "Sensing Matrix",
    "Sparsifying Matrix",
    "Algorithm",
    "Avg NMSE",
    "Std NMSE",
    "Avg PRD",
    "Std PRD",
    "Avg SNDR",
    "Std SNDR",
    "Avg Elapsed Time (s)",
    "Std Elapsed Time (s)",
  ]

  # Format the output with combined mean ± std
  formatted_stats = pd.DataFrame(
    {
      "Sensing Matrix": stats["Sensing Matrix"],
      "Sparsifying Matrix": stats["Sparsifying Matrix"],
      "Algorithm": stats["Algorithm"],
      "NMSE (mean ± std)": stats.apply(
        lambda row: f"{row['Avg NMSE']:.6f} ± {row['Std NMSE']:.6f}", axis=1
      ),
      "PRD (mean ± std)": stats.apply(
        lambda row: f"{row['Avg PRD']:.6f} ± {row['Std PRD']:.6f}", axis=1
      ),
      "SNDR (mean ± std)": stats.apply(
        lambda row: f"{row['Avg SNDR']:.6f} ± {row['Std SNDR']:.6f}", axis=1
      ),
      "Elapsed Time (mean ± std)": stats.apply(
        lambda row: f"{row['Avg Elapsed Time (s)']:.6f} ± {row['Std Elapsed Time (s)']:.6f}",
        axis=1,
      ),
    }
  )

  return formatted_stats, stats


if __name__ == "__main__":
  print("Building statistics table from database...\n")

  formatted_table, raw_stats = build_statistics_table(
    db_filename="eeg_cs_evaluations.db"
  )

  if not formatted_table.empty:
    print("=" * 150)
    print("STATISTICS TABLE (Formatted)")
    print("=" * 150)
    print(formatted_table.to_string(index=False))
    print("\n")

    print("=" * 150)
    print("STATISTICS TABLE (Separate Columns)")
    print("=" * 150)
    print(raw_stats.to_string(index=False))
    print("\n")

    # Save to CSV files
    output_formatted = Path(__file__).parent / "statistics_table_formatted.csv"
    output_raw = Path(__file__).parent / "statistics_table_raw.csv"

    formatted_table.to_csv(output_formatted, index=False)
    raw_stats.to_csv(output_raw, index=False)

    print("Tables saved to:")
    print(f"  - {output_formatted}")
    print(f"  - {output_raw}")
  else:
    print("No results to display.")
