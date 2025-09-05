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
from scipy.signal import resample


@dataclass()
class Loader(ABC):
  """
  Abstract class for data loaders.
  This class defines the interface for loading datasets.
  """

  segment_length_s: float
  max_blocks_per_file_per_run: int | None = None
  random_state: int = 42
  root_dir: str | None = None

  _current_file_path: str | None = field(init=False, default=None)
  _dataset: str = field(init=False)
  _fs: float = field(init=False)
  _ch_names: list[str] = field(init=False, default_factory=list)  # type: ignore
  _n_samples: int = field(init=False)
  _n_channels: int = field(init=False)
  _file_paths: list[str] = field(init=False, default_factory=list)  # type: ignore
  _available_possible_starts_by_file: dict[str, list[int] | None] = field(  # type: ignore
    init=False, default_factory=dict
  )

  def __post_init__(self) -> None:
    self._load_metadata()
    self._file_paths = self._load_file_paths()
    self._available_possible_starts_by_file: dict[str, list[int] | None] = (
      dict.fromkeys(self._file_paths, None)
    )
    self._n_samples = int(self.segment_length_s * self._fs)

  def blocks_generator(
    self, downsampled_fs: int | None = None
  ) -> Generator[tuple[npt.NDArray[np.float64], int, str], None, None]:
    n_samples = (
      self.get_downsampled_n_samples(downsampled_fs)
      if downsampled_fs is not None
      else self.n_samples
    )

    rng = np.random.default_rng(self.random_state)

    while self.has_available_blocks():
      rng.shuffle(self.file_paths)

      for file_path in self.file_paths:
        self._current_file_path = file_path
        possible_starts = self._available_possible_starts_by_file[file_path]

        areAllSegmentsIncluded = (
          possible_starts is not None and len(possible_starts) == 0
        )
        if areAllSegmentsIncluded:
          continue

        data = self._load_file_data(file_path)
        if data is None:
          continue

        max_start_idx = data.shape[0] - n_samples
        if max_start_idx <= 0:
          error_msg = f"Not enough data in {file_path} for segment length {self.segment_length_s} seconds."  # noqa: E501
          raise ValueError(error_msg)

        if possible_starts is None:
          n_starts = max_start_idx // n_samples + 1
          possible_starts = (n_samples * rng.permutation(n_starts)).tolist()

        added_blocks_count = 0
        max_blocks = self.max_blocks_per_file_per_run or len(possible_starts) or 0
        for start in possible_starts:
          end = start + n_samples
          block = data[start:end, :]
          if downsampled_fs is not None:
            block = resample(block, n_samples)

          yield (block, start, self._current_file_path)

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

  def get_downsampled_n_samples(self, downsampled_fs: float) -> int:
    if downsampled_fs == self.fs:
      return self.n_samples
    if downsampled_fs > self.fs:
      error_msg = (
        "New sampling frequency must be less than the current sampling frequency."
      )
      raise ValueError(error_msg)
    if downsampled_fs <= 0:
      error_msg = "New sampling frequency must be positive."
      raise ValueError(error_msg)

    return int(np.round(self.n_samples * downsampled_fs / self.fs))

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
  def fs(self) -> float:
    return self._fs

  @property
  def ch_names(self) -> list[str]:
    return self._ch_names

  @property
  def n_samples(self) -> int:
    return self._n_samples

  @property
  def n_channels(self) -> int:
    return self._n_channels

  @property
  def file_paths(self) -> list[str]:
    return self._file_paths

  @property
  def current_file_path(self) -> str | None:
    return self._current_file_path


@dataclass
class CHBMITLoader(Loader):
  """
  Loader for CHB-MIT Scalp EEG Database.
  """

  def _load_metadata(self) -> None:
    self._dataset = "chbmit"
    self._fs = 256.0
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
    self._n_samples = 5
    if self.root_dir is None:
      self.root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/CHBMIT"

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
    self._fs = 1000.0
    if self.root_dir is None:
      self.root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/BCIIV_1"

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
    self._fs = 240.0
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


if __name__ == "__main__":
  loader = CHBMITLoader(
    max_blocks_per_file_per_run=10, segment_length_s=4, random_state=42
  )

  # folder_path = "./files/processed"
  # file_path = os.path.join(
  #   folder_path,
  #   "test.pkl",
  # )
  # with open(file_path, "wb") as f:
  #   pickle.dump(loader.available_possible_starts_by_file, f)

  # loader = BCIIVLoader(n_blocks=10, segment_length_s=4, max_blocks_per_file_per_run=1)
  # loader = BCIIIILoader(
  #   n_blocks=10, segment_length_s=4, max_blocks_per_file_per_run=1
  # )
  # loader = BCIIVLoader(n_blocks=1000, segment_length_s=4.0)
  # loader = CHBMITLoader.load(f"./files/processed/1748879687102909305_chbmit_fs_128_blocks_11200.pkl")

  # loader.downsample(new_fs=128.0)  # Resample to 128 Hz

  # print(
  #   f"Loaded {loader.n_blocks} blocks with {loader.n_samples} samples and {loader.n_channels} channels each."
  # )
  # print(f"Sampling frequency: {loader.fs} Hz")
  # print(f"Channel names: {loader.ch_names}")
  # print(f"Data shape: {loader.data.shape}")
  # print(loader.get_random_segments(n_segments=10).shape)
