import scipy.io as sio
import numpy as np

class BCIIVLoaderTest:
    """
    Loader for BCI Competition IV .mat EEG files.
    Allows channel selection and time slicing.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._load_mat()

    def _load_mat(self):
        mat_data = sio.loadmat(self.file_path, spmatrix=False)
        self.fs = int(mat_data["nfo"]["fs"][0][0][0][0])
        self.ch_names = np.array([arr[0] for arr in mat_data["nfo"]["clab"][0][0][0]])

        # INT16 to V
        self.data = 1e-7 * np.array(mat_data["cnt"], dtype=np.float64) # shape: (samples, channels)
        self.n_samples, self.n_channels = self.data.shape

    def get_random_segments(self, segment_length_sec: float, n_segments: int, channels: tuple[str] | None = None, random_state: int =42):
        """
        Returns a list of randomly selected non-overlapping segments, each from a random channel (optionally filtered).
        Each segment is a 1D numpy array of signal values.
        channels: list of channel names or indices (default: all)
        """
        rng = np.random.default_rng(random_state)
        segment_samples = int(segment_length_sec * self.fs)
        max_start = self.n_samples - segment_samples

        if n_segments * segment_samples > self.n_samples:
            raise ValueError("Not enough data for the requested number of non-overlapping segments.")

        # Determine channel indices to use
        if channels is None:
            channel_indices = np.arange(self.n_channels)
        else:
            channel_indices = []
            for ch in channels:
                if isinstance(ch, str):
                    idx = np.where(self.ch_names == ch)[0]
                    if len(idx) == 0:
                        raise ValueError(f"Channel '{ch}' not found.")
                    channel_indices.append(idx[0])
                else:
                    channel_indices.append(int(ch))
            channel_indices = np.array(channel_indices)

        # All possible valid start indices for non-overlapping segments
        possible_starts = np.arange(0, max_start + 1, segment_samples)
        if len(possible_starts) < n_segments:
            raise ValueError("Not enough non-overlapping segments available.")

        # Randomly choose start indices without replacement
        chosen_starts = rng.choice(possible_starts, size=n_segments, replace=False)
        segments = []
        for start in chosen_starts:
            end = start + segment_samples
            ch_idx = rng.choice(channel_indices)  # random channel from filtered set
            segment = self.data[start:end, ch_idx]
            segments.append(segment)

        return np.array(segments, dtype=np.float64)

    def get_channel_data(self, channel):
        """
        Returns the data for a specific channel (by name or index).
        """
        if isinstance(channel, str):
            idx = np.where(self.ch_names == channel)[0]
            if len(idx) == 0:
                raise ValueError(f"Channel '{channel}' not found.")
            idx = idx[0]
        else:
            idx = int(channel)
        return self.data[:, idx]

    def get_data(self, channels=None, tmin=None, tmax=None):
        """
        Returns data for selected channels and time range.
        channels: list of channel names or indices (default: all)
        tmin, tmax: time in seconds (default: full range)
        """
        # Channel selection
        if channels is None:
            ch_idx = np.arange(self.n_channels)
        else:
            ch_idx = []
            for ch in channels:
                if isinstance(ch, str):
                    idx = np.where(self.ch_names == ch)[0]
                    if len(idx) == 0:
                        raise ValueError(f"Channel '{ch}' not found.")
                    ch_idx.append(idx[0])
                else:
                    ch_idx.append(int(ch))
            ch_idx = np.array(ch_idx)
        
        # Time selection
        start = 0 if tmin is None else int(self.fs * tmin)
        end = self.n_samples if tmax is None else int(self.fs * tmax)

        # Clamp to valid range
        start = max(0, start)
        end = min(self.n_samples, end)
        
        t = np.arange(start, end) / self.fs
        X = self.data[start:end, ch_idx]
    
        # Return data [samples, channels], time vector and sampling frequency
        return X, t, self.fs
