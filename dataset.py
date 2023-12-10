#!/usr/bin/env python3

import numpy as np
from torch.utils.data import Dataset
import random


class IoanninaEEGDataset(Dataset):
    def __init__(
        self,
        eeg_data_raw_path: str,
        eeg_data_clean_path: str,
        eeg_count: int = -1,
    ):
        """Dataset for EEG noise removal.

        Data expected dimensions:
        epochs_count x channels_count x sample_len
        """
        self._eeg_data_raw = np.load(eeg_data_raw_path).astype(np.float32)
        self._eeg_data_clean = np.load(eeg_data_clean_path).astype(np.float32)
        assert (
            self._eeg_data_raw.shape == self._eeg_data_clean.shape
        ), "Both datasets must have the same shape (epochs_count x channels_count x sample_len)"

        if eeg_count > 0:
            if eeg_count > self.__len__():
                raise ValueError(
                    f"eeg_count ({eeg_count}) must be less than epochs count {self.__len__()}"
                )

            epoch_indexes = np.random.choice(self.__len__(), eeg_count, replace=False)
            self._eeg_data_raw = self._eeg_data_raw[epoch_indexes]
            self._eeg_data_clean = self._eeg_data_clean[epoch_indexes]

    def __len__(self) -> int:
        return len(self._eeg_data_raw)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self._eeg_data_raw[index], self._eeg_data_clean[index]
