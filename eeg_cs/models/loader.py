import glob
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field

import mne
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from scipy.signal import resample_poly


@dataclass
class Loader(ABC):
  """
  Abstract class for data loaders.
  This class defines the interface for loading datasets.
  """

  root_dir: str | None = None

  _current_file_path: str | None = field(init=False, default=None)
  _dataset: str = field(init=False)
  _fs: int = field(init=False)
  _ch_names: list[str] = field(init=False, default_factory=list)  # type: ignore
  _n_samples: int = field(init=False)
  _n_channels: int = field(init=False)
  _file_paths: list[str] = field(init=False, default_factory=list)  # type: ignore
  _available_possible_starts_by_file: dict[str, list[int] | None] = field(  # type: ignore
    init=False, default_factory=dict
  )
  _name: str = field(init=False)

  def __post_init__(self) -> None:
    self._load_metadata()
    self._file_paths = self._load_file_paths()
    self._available_possible_starts_by_file: dict[str, list[int] | None] = (
      dict.fromkeys(self._file_paths, None)
    )

  def get_signal(
    self,
    fileName: str,
    start_time_idx: int = 0,
    segment_length_s: float | None = None,
    downsampled_fs: float | None = None,
    ch_name: str | None = None,
  ) -> npt.NDArray[np.float64]:
    if start_time_idx < 0:
      raise ValueError("Start time index must be non-negative")

    if segment_length_s is not None and segment_length_s <= 0.0:
      raise ValueError("Segment length must be positive")

    ch_idx = None
    if ch_name is not None:
      if ch_name not in self.ch_names:
        raise ValueError(f"Channel '{ch_name}' not found")

      ch_idx = self.ch_names.index(ch_name)

    file_path = next(
      (path for path in self.file_paths if fileName in [path, os.path.basename(path)]),
      None,
    )
    if file_path is None:
      raise FileNotFoundError(f"File '{fileName}' not found in loaded file paths")

    data = self._load_file_data(file_path)
    if data is None:
      raise ValueError(f"Could not load data from file '{fileName}'")

    if start_time_idx >= data.shape[0]:
      raise ValueError("Start time index exceeds signal length")

    remaining_samples = data.shape[0] - start_time_idx
    segment_length_samples = (
      min(int(segment_length_s * self.fs), remaining_samples)
      if segment_length_s
      else remaining_samples
    )

    data = data[start_time_idx : (start_time_idx + segment_length_samples), :]

    if ch_idx is not None:
      data = data[:, ch_idx]

    if downsampled_fs is not None:
      if downsampled_fs <= 0.0:
        raise ValueError("Downsampled frequency must be positive")
      if downsampled_fs > self.fs:
        raise ValueError(
          "Downsampled frequency must be less than or equal to original frequency"
        )

      data = resample_poly(
        data, up=int(downsampled_fs), down=int(self.fs), axis=0
      ).astype(np.float64)

    return data

  def blocks_generator(
    self,
    segment_length_s: float,
    max_blocks_per_file_per_run: int | None = None,
    downsampled_fs: int | None = None,
    random_state: int | None = None,
  ) -> Generator[tuple[npt.NDArray[np.float64], int, str], None, None]:
    n_samples_original = int(segment_length_s * self.fs)

    rng = np.random.default_rng(random_state)

    while self.has_available_blocks():
      rng.shuffle(self.file_paths)

      for file_path in self.file_paths:
        self._current_file_path = file_path
        file_name = os.path.basename(file_path)

        possible_starts = self._available_possible_starts_by_file[file_path]

        areAllSegmentsIncluded = (
          possible_starts is not None and len(possible_starts) == 0
        )
        if areAllSegmentsIncluded:
          continue

        data = self._load_file_data(file_path)
        if data is None:
          continue

        max_start_idx = data.shape[0] - n_samples_original
        if max_start_idx <= 0:
          error_msg = f"Not enough data in {file_path} for segment length {segment_length_s} seconds."  # noqa: E501
          raise ValueError(error_msg)

        if possible_starts is None:
          n_starts = max_start_idx // n_samples_original + 1
          possible_starts = (n_samples_original * rng.permutation(n_starts)).tolist()

        added_blocks_count = 0
        max_blocks = max_blocks_per_file_per_run or len(possible_starts) or 0
        for start in possible_starts:
          end = start + n_samples_original
          block = data[start:end, :]

          # # Artifact removal: skip blocks with max amplitude > 150 µV
          # if block.max() > 150e-6:
          #   print("Artifact detected, skipping block.")
          #   continue

          if downsampled_fs is not None and downsampled_fs != self.fs:
            block = resample_poly(
              block, up=downsampled_fs, down=self.fs, axis=0
            ).astype(np.float64)

          yield (block, start, file_name)

          added_blocks_count += 1
          if added_blocks_count == max_blocks:
            break

        self._available_possible_starts_by_file[file_path] = possible_starts[
          added_blocks_count:
        ]

  def has_available_blocks(self) -> bool:
    return any(
      possible_starts is None or len(possible_starts) > 0
      for possible_starts in self._available_possible_starts_by_file.values()
    )

  def get_downsampled_n_samples(
    self, segment_length_s: float, downsampled_fs: float
  ) -> int:
    if downsampled_fs == self.fs:
      return int(segment_length_s * self.fs)
    if downsampled_fs > self.fs:
      raise ValueError(
        "New sampling frequency must be less than the current sampling frequency."
      )
    if downsampled_fs <= 0:
      raise ValueError("New sampling frequency must be positive.")

    return int(np.round(segment_length_s * downsampled_fs))

  def save(self, file_path: str) -> None:
    with open(file_path, "wb") as f:
      pickle.dump(self, f)

  @classmethod
  def load(cls, file_path: str) -> "Loader":
    obj = None
    with open(file_path, "rb") as f:
      obj = pickle.load(f)

    if not isinstance(obj, cls):
      raise TypeError(f"Loaded object is not of type {cls.__name__}")

    return obj

  @abstractmethod
  def _load_metadata(self) -> None: ...

  @abstractmethod
  def _load_file_data(self, file_path: str) -> npt.NDArray[np.float64] | None: ...

  @abstractmethod
  def _load_file_paths(self) -> list[str]: ...

  @property
  def dataset(self) -> str:
    return self._dataset

  @property
  def fs(self) -> int:
    return self._fs

  @property
  def ch_names(self) -> list[str]:
    return self._ch_names

  @property
  def n_channels(self) -> int:
    return self._n_channels

  @property
  def file_paths(self) -> list[str]:
    return self._file_paths

  @property
  def current_file_path(self) -> str | None:
    return self._current_file_path

  @property
  def name(self) -> str:
    return self._name


@dataclass
class CHBMITLoader(Loader):
  """
  Loader for CHB-MIT Scalp EEG Database.
  """

  def _load_metadata(self) -> None:
    self._dataset = "chbmit"
    self._fs = 256
    self._ch_names = [
      "FP1-F7",
      "F7-T7",
      "T7-P7",
      "P7-O1",
      "FP1-F3",
      "F3-C3",
      "C3-P3",
      "P3-O1",
      "FP2-F4",
      "F4-C4",
      "C4-P4",
      "P4-O2",
      "FP2-F8",
      "F8-T8",
      "T8-P8",
      "P8-O2",
      "FZ-CZ",
      "CZ-PZ",
      "P7-T7",
      "T7-FT9",
      "FT9-FT10",
      "FT10-T8",
    ]
    self._n_channels = len(self.ch_names)
    if self.root_dir is None:
      self.root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/CHBMIT"
    self._name = "CHB-MIT"

  def _load_file_data(self, file_path: str) -> npt.NDArray[np.float64] | None:
    raw = mne.io.read_raw_edf(file_path, include=self.ch_names, verbose=False)
    if len(raw.info["ch_names"]) == 0:
      return None

    # (n_samples, n_channels)
    data = raw.get_data().T[:, :-1]

    data_n_channels = data.shape[1]
    if data_n_channels != self.n_channels:
      return None

    return data

  def _load_file_paths(self) -> list[str]:
    edf_file_paths = sorted(glob.glob(os.path.join(self.root_dir, "chb*/", "*.edf")))
    if not edf_file_paths:
      error_msg = f"No EDF files found in {self.root_dir}."
      raise FileNotFoundError(error_msg)

    return edf_file_paths


@dataclass
class BCIIVLoader(Loader):
  """
  Loader for BCI IV dataset I.
  """

  def _load_metadata(self) -> None:
    self._dataset = "bciiv_1"
    self._fs = 1000
    if self.root_dir is None:
      self.root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/BCIIV_1"
    self._name = "BCI IV"

  def _load_file_data(self, file_path: str) -> npt.NDArray[np.float64] | None:
    mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    self._ch_names = mat_data["nfo"].clab
    self._n_channels = len(self._ch_names)

    # INT16 to V
    # shape: (samples, channels)
    return 1e-7 * np.array(mat_data["cnt"], dtype=np.float64)

  def _load_file_paths(self) -> list[str]:
    mat_file_paths = sorted(glob.glob(os.path.join(self.root_dir, "*.mat")))
    if not mat_file_paths:
      error_msg = f"No .mat files found in {self.root_dir}."
      raise FileNotFoundError(error_msg)

    return mat_file_paths


class BCIIIILoader(Loader):
  """
  Loader for BCI III dataset II.
  """

  def _load_metadata(self) -> None:
    self._dataset = "bciiii_2"
    self._fs = 240
    self._ch_names = [
      "FC5",
      "FC3",
      "FC1",
      "FCz",
      "FC2",
      "FC4",
      "FC6",
      "C5",
      "C3",
      "C1",
      "Cz",
      "C2",
      "C4",
      "C6",
      "CP5",
      "CP3",
      "CP1",
      "CPz",
      "CP2",
      "CP4",
      "CP6",
      "Fp1",
      "Fpz",
      "Fp2",
      "AF7",
      "AF3",
      "AFz",
      "AF4",
      "AF8",
      "F7",
      "F5",
      "F3",
      "F1",
      "Fz",
      "F2",
      "F4",
      "F6",
      "F8",
      "FT7",
      "FT8",
      "T7",
      "T8",
      "T9",
      "T10",
      "TP7",
      "TP8",
      "P7",
      "P5",
      "P3",
      "P1",
      "Pz",
      "P2",
      "P4",
      "P6",
      "P8",
      "PO7",
      "PO3",
      "POz",
      "PO4",
      "PO8",
      "O1",
      "Oz",
      "O2",
      "Iz",
    ]
    self._n_channels = len(self._ch_names)

    if self.root_dir is None:
      self.root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/BCIIII_2"
    self._name = "BCIII"

  def _load_file_data(self, file_path: str) -> npt.NDArray[np.float64] | None:
    mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # Extract EEG signal: expected shape (epochs, samples, channels)
    data = 1e-6 * mat_data["Signal"].astype(np.float64)

    if data.ndim == 3:
      epochs, samples, channels = data.shape
      data = data.reshape(epochs * samples, channels)

    return data

  def _load_file_paths(self) -> list[str]:
    mat_file_paths = sorted(glob.glob(os.path.join(self.root_dir, "*.mat")))

    if not mat_file_paths:
      error_msg = f"No .mat files found in {self.root_dir}."
      raise FileNotFoundError(error_msg)

    return mat_file_paths
