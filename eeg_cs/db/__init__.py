"""Database utilities for eeg_cs.

Exports:
  - SQLiteClient: simple SQLite helper to init/reset DB and run queries.
"""

from .client import SQLiteClient

__all__ = ["SQLiteClient"]
