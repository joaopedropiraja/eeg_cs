import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import mne
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from scipy.signal import resample


@dataclass
class Loader(ABC):
  """
  Abstract class for data loaders.
  This class defines the interface for loading datasets.
  """

  # Number of blocks of segments in the dataset
  n_blocks: int
  segment_length_sec: float
  max_blocks_per_file_per_run: int | None = None
  root_dir: str | None = None
  random_state: int = 42

  _dataset: str = field(init=False)
  _fs: float = field(init=False)
  _ch_names: list[str] = field(init=False, default_factory=list)

  # (n_blocks, n_samples, n_channels)
  _data: npt.NDArray[np.float64] = field(
    init=False, default_factory=lambda: np.array([], dtype=np.float64)
  )

  _n_samples: int = field(init=False)
  _n_channels: int = field(init=False)

  def __post_init__(self) -> None:
    self._load_metadata()
    self._load_data()

  def _load_data(self) -> None:
    file_paths = self._load_file_paths()
    n_samples_per_segment = int(self.segment_length_sec * self._fs)
    blocks = self._select_blocks(file_paths, n_samples_per_segment)

    if len(blocks) < self.n_blocks:
      error_msg = (
        f"Requested {self.n_blocks} blocks, but only {len(blocks)} were loaded."
      )
      raise ValueError(error_msg)

    # Shape (n_blocks, n_samples_per_segment, n_channels)
    self._data = np.array(blocks)
    self._n_samples = n_samples_per_segment

  def _select_blocks(
    self, file_paths: list[str], n_samples_per_segment: int
  ) -> list[npt.NDArray[np.float64]]:
    rng = np.random.default_rng(self.random_state)

    available_possible_starts_by_file: dict[str, list[int] | None] = dict.fromkeys(
      file_paths
    )
    blocks: list[npt.NDArray[np.float64]] = []

    while len(blocks) < self.n_blocks:
      rng.shuffle(file_paths)

      for file_path in file_paths:
        possible_starts = available_possible_starts_by_file[file_path]

        areAllSegmentsIncluded = (
          possible_starts is not None and len(possible_starts) == 0
        )
        if areAllSegmentsIncluded:
          continue

        data = self._load_file_data(file_path)
        if data is None:
          continue

        max_start_idx = data.shape[0] - n_samples_per_segment
        if max_start_idx <= 0:
          error_msg = f"Not enough data in {file_path} for segment length {self.segment_length_sec} seconds."
          raise ValueError(error_msg)

        if possible_starts is None:
          possible_starts = np.arange(0, max_start_idx, n_samples_per_segment)
          rng.shuffle(possible_starts)

        added_blocks_count = 0
        while possible_starts.size > 0:
          start, possible_starts = possible_starts[0], np.delete(possible_starts, 0)
          end = start + n_samples_per_segment

          block = data[start:end, :]
          blocks.append(block)

          if len(blocks) == self.n_blocks:
            return blocks

          added_blocks_count += 1
          if (
            self.max_blocks_per_file_per_run is not None
            and added_blocks_count == self.max_blocks_per_file_per_run
          ):
            break

        available_possible_starts_by_file[file_path] = possible_starts

    return blocks

  @abstractmethod
  def _load_metadata(self) -> None: ...

  @abstractmethod
  def _load_file_data(self, file_path: str) -> npt.NDArray[np.float64] | None: ...

  @abstractmethod
  def _load_file_paths(self) -> list[str]: ...

  def get_random_segments(
    self, n_segments: int, random_state: int = 42
  ) -> npt.NDArray[np.float64]:
    """
    Returns a list of individual segments randomly selected from the dataset.
    """
    rng = np.random.default_rng(random_state)

    total_n_segments = self.n_blocks * self.n_channels

    if n_segments > total_n_segments:
      error_msg = (
        f"Requested {n_segments} segments, but only {total_n_segments} available."
      )
      raise ValueError(error_msg)

    indices = rng.choice(total_n_segments, size=n_segments, replace=False)

    segments: list[npt.NDArray[np.float64]] = []
    for idx in indices:
      block_idx = idx // self.n_channels
      channel_idx = idx % self.n_channels
      segments.append(self._data[block_idx, :, channel_idx])

    return np.array(segments).T

  def downsample(self, new_fs: float) -> None:
    """
    Resample the data to a new sampling frequency.
    Assumes self._data has shape (n_blocks, n_samples, n_channels).
    """
    error_msg: str = ""

    if self._fs == new_fs:
      return
    if new_fs > self._fs:
      error_msg = (
        "New sampling frequency must be less than the current sampling frequency."
      )
      raise ValueError(error_msg)
    if new_fs <= 0:
      error_msg = "New sampling frequency must be positive."
      raise ValueError(error_msg)

    new_n_samples = int(np.round(self.n_samples * new_fs / self._fs))
    # Resample each block independently along the samples axis
    self._data = resample(self._data, new_n_samples, axis=1)
    self._n_samples = new_n_samples
    self._fs = new_fs

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
  def data(self) -> npt.NDArray[np.float64]:
    return self._data

  @property
  def n_samples(self) -> int:
    return self._n_samples

  @property
  def n_channels(self) -> int:
    return self._n_channels

  @property
  def n_blocks(self) -> int:
    return self._n_blocks

  @n_blocks.setter
  def n_blocks(self, n_blocks: int) -> None:
    self._n_blocks = n_blocks

  @property
  def segment_length_sec(self) -> float:
    return self._segment_length_sec

  @segment_length_sec.setter
  def segment_length_sec(self, segment_length_sec: int) -> None:
    self._segment_length_sec = segment_length_sec


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
    n_blocks=10, segment_length_sec=4, max_blocks_per_file_per_run=1
  )
  loader = BCIIVLoader(n_blocks=10, segment_length_sec=4, max_blocks_per_file_per_run=1)
  loader = BCIIIILoader(
    n_blocks=10, segment_length_sec=4, max_blocks_per_file_per_run=1
  )
  # loader = BCIIVLoader(n_blocks=1000, segment_length_sec=4.0)
  # loader = CHBMITLoader.load(f"./files/processed/1748879687102909305_chbmit_fs_128_blocks_11200.pkl")

  # loader.downsample(new_fs=128.0)  # Resample to 128 Hz

  print(
    f"Loaded {loader.n_blocks} blocks with {loader.n_samples} samples and {loader.n_channels} channels each."
  )
  print(f"Sampling frequency: {loader.fs} Hz")
  print(f"Channel names: {loader.ch_names}")
  print(f"Data shape: {loader.data.shape}")
  # print(loader.get_random_segments(n_segments=10).shape)
