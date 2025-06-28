import glob
import os
import pickle
import time
from abc import ABC, abstractmethod

import mne
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from scipy.signal import resample


class Loader(ABC):
    """
    Abstract class for data loaders.
    This class defines the interface for loading datasets.
    """

    _id: int = time.time_ns()  # Unique identifier for the loader instance
    _dataset: str
    _fs: float
    _ch_names: list[str]
    _data: npt.NDArray[np.float32]  # (n_blocks, n_samples, n_channels)
    _n_samples: int
    _n_channels: int
    # Number of blocks of segments in the dataset
    _n_blocks: int
    _segment_length_sec: float

    def __init__(
        self, n_blocks: int, segment_length_sec: float, random_state: int = 42
    ) -> None:
        self._n_blocks = n_blocks
        self._segment_length_sec = segment_length_sec

        self._load_data(n_blocks, segment_length_sec, random_state)

    @abstractmethod
    def _load_data(
        self, n_blocks: int, segment_length_sec: float, random_state: int
    ) -> None: ...

    def get_random_segments(
        self, n_segments: int, random_state: int = 42
    ) -> npt.NDArray[np.float32]:
        """
        Returns a list of individual segments randomly selected from the dataset.
        """
        rng = np.random.default_rng(random_state)

        total_n_segments = self.n_blocks * self.n_channels

        if n_segments > total_n_segments:
            raise ValueError(
                f"Requested {n_segments} segments, but only {total_n_segments} available."
            )

        # Generate random indices for blocks and channels
        indices = rng.choice(total_n_segments, size=n_segments, replace=False)

        # Map indices to block and channel
        segments: list[npt.NDArray[np.float32]] = []
        for idx in indices:
            block_idx = idx // self.n_channels
            channel_idx = idx % self.n_channels
            segments.append(self._data[block_idx, :, channel_idx])

        return np.array(segments).T

    # def split_training_and_test(
    #     self, n_blocks: int, flatten: bool = False, remove_mean: bool = False
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Split the dataset into training and test sets.
    #     The first n_blocks are used for training, and the rest for testing.

    #     Parameters
    #     ----------
    #     n_blocks : int
    #         Number of blocks to use for training.
    #     flatten : bool
    #         If True, flatten the training data to shape (n_samples, n_blocks * n_channels).
    #     remove_mean : bool
    #         If True, remove the mean value from each channel in each block of the training data.

    #     Returns
    #     -------
    #     training_data : np.ndarray
    #         If flatten=False, shape is (n_blocks, n_samples, n_channels).
    #         If flatten=True, shape is (n_samples, n_blocks * n_channels).
    #     test_data : np.ndarray
    #         Shape is (remaining_blocks, n_samples, n_channels).
    #     """
    #     if n_blocks > self.n_blocks:
    #         raise ValueError(
    #             f"Requested {n_blocks} blocks, but only {self.n_blocks} available."
    #         )
    #     if n_blocks <= 0:
    #         raise ValueError("Number of blocks must be positive.")

    #     # Select the first n_blocks for training and the rest for testing
    #     training_data = self._data[
    #         :n_blocks
    #     ]  # shape: (n_blocks, n_samples, n_channels)
    #     test_data = self._data[
    #         n_blocks:
    #     ]  # shape: (remaining_blocks, n_samples, n_channels)

    #     # Work on a copy so we don't alter the original dataset
    #     training_data = training_data.copy()

    #     # If requested, remove the mean from each channel in each block
    #     if remove_mean:
    #         # Subtract mean over samples for each (block, channel)
    #         # training_data has shape (n_blocks, n_samples, n_channels)
    #         means = training_data.mean(
    #             axis=1, keepdims=True
    #         )  # (n_blocks, 1, n_channels)
    #         training_data = training_data - means

    #     if flatten:
    #         # We want shape (n_samples, n_blocks * n_channels).
    #         # Current training_data is (n_blocks, n_samples, n_channels).
    #         # First transpose to (n_samples, n_blocks, n_channels), then reshape.
    #         segments = []
    #         for block_idx in range(n_blocks):
    #             for channel_idx in range(self.n_channels):
    #                 segment = training_data[block_idx, :, channel_idx]
    #                 segments.append(segment)

    #         training_data = np.array(segments).T
    #         # n_samples = self.n_samples
    #         # n_channels = self.n_channels
    #         # training_data = training_data.transpose(1, 0, 2)  # (n_samples, n_blocks, n_channels)
    #         # training_data = training_data.reshape(n_samples, n_blocks * n_channels)

    #     return training_data, test_data

    def downsample(self, new_fs: float) -> None:
        """
        Resample the data to a new sampling frequency.
        Assumes self._data has shape (n_blocks, n_samples, n_channels).
        """
        if self._fs == new_fs:
            return
        if new_fs > self._fs:
            raise ValueError(
                "New sampling frequency must be less than the current sampling frequency."
            )
        if new_fs <= 0:
            raise ValueError("New sampling frequency must be positive.")

        new_n_samples = int(np.round(self.n_samples * new_fs / self._fs))
        # Resample each block independently along the samples axis
        self._data = resample(self._data, new_n_samples, axis=1)
        self._n_samples = new_n_samples
        self._fs = new_fs

    @classmethod
    def load(cls, file_path: str) -> "Loader":
        """
        Load data from a file.
        This method should be implemented by subclasses.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        with open(file_path, "rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(
                    f"Expected object of type {cls.__name__}, got {type(obj).__name__}."
                )

            return obj

    def save(self, folder_path: str) -> None:
        file_path = os.path.join(
            folder_path,
            f"{self.id}_{self.dataset}_fs_{self.fs}_blocks_{self.n_blocks}.pkl",
        )
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @property
    def id(self) -> int:
        return self._id

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
    def data(self) -> npt.NDArray[np.float32]:
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

    @property
    def segment_length_sec(self) -> float:
        return self._segment_length_sec


class CHBMITLoader(Loader):
    """
    Loader for CHB-MIT Scalp EEG Database.
    """

    _dataset = "chbmit"
    _fs = 256.0  # Default sampling frequency for CHB-MIT
    _ch_names = [
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
    _n_channels = len(_ch_names)

    def _load_data(self, n_blocks: int, segment_length_sec: float, random_state: int):
        """
        Load EEG data from the CHB-MIT dataset.

        Parameters
        ----------
        n_blocks : int
            Number of blocks to load.
        segment_length_sec : float
            Length of each segment in seconds.
        random_state : int
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(random_state)
        root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/CHBMIT"
        # root_dir = f"./../files/CHBMIT"
        edf_files = sorted(glob.glob(os.path.join(root_dir, "chb*/", "*.edf")))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found in {root_dir}.")

        n_samples_per_segment = int(segment_length_sec * self._fs)
        blocks: list[npt.NDArray[np.float32]] = []

        # Keep track of the last used index for each EDF file
        used_indice: dict[str, int | None] = dict.fromkeys(edf_files)
        max_blocks_per_file = max(1, n_blocks // 10)

        while len(blocks) < n_blocks:
            old_blocks_count = len(blocks)
            rng.shuffle(edf_files)

            for edf_file in edf_files:
                areAllSegmentsIncluded = used_indice[edf_file] == 0
                if areAllSegmentsIncluded:
                    continue

                raw = mne.io.read_raw_edf(
                    edf_file, include=self.ch_names, verbose=False
                )
                if len(raw.info["ch_names"]) == 0:
                    continue

                data = raw.get_data().T[:, :-1]  # (n_samples, n_channels)

                data_n_channels = data.shape[1]
                if data_n_channels != self.n_channels:
                    continue

                max_start_idx = data.shape[0] - n_samples_per_segment
                if max_start_idx <= 0:
                    raise ValueError(
                        f"Not enough data in {edf_file} for segment length {segment_length_sec} seconds."
                    )

                end_idx = (
                    used_indice[edf_file]
                    if used_indice[edf_file] is not None
                    else max_start_idx + 1
                )
                possible_starts: np.ndarray = np.arange(
                    0, end_idx, n_samples_per_segment
                )

                initial_idx = rng.choice(possible_starts.size)
                used_indice[edf_file] = initial_idx

                # Skip if the initial index is 0, as it would not yield new blocks
                if initial_idx == 0:
                    continue

                added_blocks_count = 0
                for start in possible_starts[initial_idx:]:
                    end = start + n_samples_per_segment
                    block = data[start:end, :]
                    blocks.append(block)
                    added_blocks_count += 1

                    if (
                        len(blocks) == n_blocks
                        or added_blocks_count == max_blocks_per_file
                    ):
                        break

                if len(blocks) == n_blocks:
                    break

            # If no new blocks were added in this pass, break to avoid infinite loop
            if len(blocks) == old_blocks_count:
                break

        if len(blocks) < n_blocks:
            raise ValueError(
                f"Requested {n_blocks} blocks, but only {len(blocks)} were loaded."
            )

        self._data = np.array(
            blocks
        )  # Shape (n_blocks, n_samples_per_segment, n_channels)
        self._n_samples = n_samples_per_segment


class BCIIVLoader(Loader):
    """
    Loader for BCI IV dataset I.
    """

    _dataset = "bciiv_1"
    _fs = 1000.0  # Default sampling frequency for BCI IV dataset I

    def _load_data(self, n_blocks: int, segment_length_sec: float, random_state: int):
        rng = np.random.default_rng(random_state)
        root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/BCIIV_1"
        mat_files = sorted(glob.glob(os.path.join(root_dir, "*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {root_dir}.")

        n_samples_per_segment = int(segment_length_sec * self._fs)
        blocks: list[np.ndarray] = []

        # Keep track of the last used index for each Mat file
        used_indice: dict[str, int | None] = dict.fromkeys(mat_files)

        while len(blocks) < n_blocks:
            old_blocks_count = len(blocks)
            rng.shuffle(mat_files)

            for mat_file in mat_files:
                areAllSegmentsIncluded = used_indice[mat_file] == 0
                if areAllSegmentsIncluded:
                    continue

                mat_data = sio.loadmat(
                    mat_file, squeeze_me=True, struct_as_record=False
                )
                self._ch_names = mat_data["nfo"].clab
                self._n_channels = len(self._ch_names)

                # INT16 to V
                data = 1e-7 * np.array(
                    mat_data["cnt"], dtype=np.double
                )  # shape: (samples, channels)

                max_start_idx = data.shape[0] - n_samples_per_segment
                if max_start_idx <= 0:
                    raise ValueError(
                        f"Not enough data in {mat_file} for segment length {segment_length_sec} seconds."
                    )

                end_idx = (
                    used_indice[mat_file]
                    if used_indice[mat_file] is not None
                    else max_start_idx + 1
                )
                possible_starts: np.ndarray = np.arange(
                    0, end_idx, n_samples_per_segment
                )

                initial_idx = rng.choice(possible_starts.size)
                used_indice[mat_file] = initial_idx

                # Skip if the initial index is 0, as it would not yield new blocks
                if initial_idx == 0:
                    continue

                for start in possible_starts[initial_idx:]:
                    end = start + n_samples_per_segment
                    block = data[start:end, :]
                    blocks.append(block)

                    if len(blocks) == n_blocks:
                        break

                if len(blocks) == n_blocks:
                    break

            # If no new blocks were added in this pass, break to avoid infinite loop
            if len(blocks) == old_blocks_count:
                break

        if len(blocks) < n_blocks:
            raise ValueError(
                f"Requested {n_blocks} blocks, but only {len(blocks)} were loaded."
            )

        self._data = np.array(
            blocks
        )  # Shape (n_blocks, n_samples_per_segment, n_channels)
        self._n_samples = n_samples_per_segment


class BCIIIILoader(Loader):
    """
    Loader for BCI III dataset II.
    """

    _dataset = "bciiii_2"
    _fs = 240.0  # Default sampling frequency for BCI III dataset II
    _ch_names = [
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
    _n_channels = len(_ch_names)

    def _load_data(self, n_blocks: int, segment_length_sec: float, random_state: int):
        rng = np.random.default_rng(random_state)
        root_dir = "/home/jplp/Disco X/UFMG/2025/TCC/Códigos/eeg_cs/files/BCIIII_2"
        mat_files = sorted(glob.glob(os.path.join(root_dir, "*.mat")))

        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {root_dir}.")

        n_samples_per_segment = int(segment_length_sec * self._fs)
        blocks: list[np.ndarray] = []

        # Iterate over .mat files in randomized order
        for mat_file in rng.permutation(mat_files):
            print(mat_file)
            mat_data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

            # Extract EEG signal: expected shape (epochs, samples, channels)
            signal = 1e-6 * mat_data["Signal"].astype(np.double)
            n_epochs, total_samples, _ = signal.shape

            # Shuffle epoch order
            epoch_indices = rng.permutation(n_epochs)
            for epoch_idx in epoch_indices:
                epoch_data = signal[epoch_idx, :, :]  # (samples, channels)

                max_start_idx = total_samples - n_samples_per_segment
                if max_start_idx < 0:
                    continue

                # Determine possible start indices within this epoch
                possible_starts = np.arange(0, max_start_idx + 1, n_samples_per_segment)
                rng.shuffle(possible_starts)

                for start_idx in possible_starts:
                    end_idx = start_idx + n_samples_per_segment
                    segment = epoch_data[start_idx:end_idx, :]  # (samples, channels)
                    blocks.append(segment)

                    if len(blocks) == n_blocks:
                        break
                if len(blocks) == n_blocks:
                    break
            if len(blocks) == n_blocks:
                break

        if len(blocks) < n_blocks:
            raise ValueError(
                f"Requested {n_blocks} blocks, but only {len(blocks)} were loaded."
            )

        # Shape: (n_blocks, n_samples_per_segment, n_channels)
        self._data = np.array(blocks)
        self._n_samples = n_samples_per_segment


if __name__ == "__main__":
    loader = BCIIIILoader(n_blocks=1000, segment_length_sec=4.0)
    # loader = BCIIVLoader(n_blocks=1000, segment_length_sec=4.0)
    # loader = CHBMITLoader(n_blocks=11200, segment_length_sec=4.0)
    # loader = CHBMITLoader.load(f"./files/processed/1748879687102909305_chbmit_fs_128_blocks_11200.pkl")

    loader.downsample(new_fs=128.0)  # Resample to 128 Hz

    print(
        f"Loaded {loader.n_blocks} blocks with {loader.n_samples} samples and {loader.n_channels} channels each."
    )
    print(f"Sampling frequency: {loader.fs} Hz")
    print(f"Channel names: {loader.ch_names}")
    print(f"Data shape: {loader.data.shape}")
    print(loader.get_random_segments(n_segments=10).shape)

    # loader.save(folder_path=f"{cwd}/../../files/processed")
