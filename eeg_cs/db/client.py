from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Iterable, Mapping

type Evaluation = tuple[
  str,
  int,
  str,
  str,
  int,
  str,
  str,
  str,
  float,
  float,
  float,
  float,
  float,
  float,
  int,
  int,
]


TABLE_NAME = "evaluations"


@dataclass
class SQLiteClient:
  db_filename: Path | str | None = None
  schema_filename: Path | str | None = None

  db_path: Path = field(init=False)
  db_schema: Path = field(init=False)

  def __post_init__(self) -> None:
    base_dir = Path(__file__).parent

    if self.db_filename is None:
      self.db_path = base_dir / "db.sqlite3"
    else:
      self.db_path = base_dir / Path(self.db_filename)

    if self.schema_filename is None:
      self.schema_path = base_dir / "init.sql"
    else:
      self.schema_path = Path(self.schema_filename)

    self.db_path.parent.mkdir(parents=True, exist_ok=True)

  def connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(str(self.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

  def create_schema(self) -> None:
    script = Path(str(self.schema_path)).read_text(encoding="utf-8")
    with self.connect() as conn:
      conn.executescript(script)
      conn.commit()
      print(f"Database schema created at {self.db_path}")

  def reset(self) -> None:
    with self.connect() as conn:
      tables = [
        r[0]
        for r in conn.execute(
          "SELECT name FROM sqlite_master "
          "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
      ]
      for t in tables:
        conn.execute(f"DROP TABLE IF EXISTS {t};")
      conn.commit()

    self.create_schema()

  def execute(
    self,
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
  ) -> int:
    with self.connect() as conn:
      cur = conn.execute(sql, params or [])
      conn.commit()
      return cur.rowcount

  def executemany(
    self,
    sql: str,
    seq_of_params: Iterable[Sequence[Any] | Mapping[str, Any]],
  ) -> int:
    with self.connect() as conn:
      cur = conn.executemany(sql, seq_of_params)
      conn.commit()
      return cur.rowcount

  def query_all(
    self,
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
  ) -> list[dict[str, Any]]:
    with self.connect() as conn:
      rows = conn.execute(sql, params or []).fetchall()
      return [dict(r) for r in rows]

  def query_one(
    self,
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
  ) -> dict[str, Any] | None:
    with self.connect() as conn:
      row = conn.execute(sql, params or []).fetchone()
      return dict(row) if row else None

  def executescript(self, script: str) -> None:
    with self.connect() as conn:
      conn.executescript(script)
      conn.commit()

  def insert_evaluation(self, row: Evaluation) -> int:
    sql = f"""
            INSERT INTO {TABLE_NAME} (
                dataset, fs, channel, file_name, start_time_idx, sensing_matrix,
                sparsifying_matrix, algorithm, mean_freq, prd, nmse, sndr, ssim, elapsed_time_s,
                compression_rate, segment_length_s
            ) VALUES (
                :dataset, :fs, :channel, :file_name, :start_time_idx, :sensing_matrix,
                :sparsifying_matrix, :algorithm, :mean_freq, :prd, :nmse, :sndr, :ssim, :elapsed_time_s,
                :compression_rate, :segment_length_s
            );
            """

    return self.execute(sql, row)

  def bulk_insert_evaluations(self, rows: Iterable[Evaluation]) -> int:
    sql = f"""
            INSERT INTO {TABLE_NAME} (
                dataset, fs, channel, file_name, start_time_idx, sensing_matrix,
                sparsifying_matrix, algorithm, mean_freq, prd, nmse, sndr, ssim, elapsed_time_s,
                compression_rate, segment_length_s
            ) VALUES (
                :dataset, :fs, :channel, :file_name, :start_time_idx, :sensing_matrix,
                :sparsifying_matrix, :algorithm, :mean_freq, :prd, :nmse, :sndr, :ssim, :elapsed_time_s,
                :compression_rate, :segment_length_s
            );
            """

    return self.executemany(sql, rows)
