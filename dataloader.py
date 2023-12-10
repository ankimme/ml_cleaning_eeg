import torch
from dataset import IoanninaEEGDataset


def create_dataloader(batch_size):
    ds = IoanninaEEGDataset(
        eeg_data_raw_path="data_npy/raw_data_train_nonorm.npy",
        eeg_data_clean_path="data_npy/clean_data_train_nonorm.npy",
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )

    ds = IoanninaEEGDataset(
        eeg_data_raw_path="data_npy/raw_data_validation_nonorm.npy",
        eeg_data_clean_path="data_npy/clean_data_validation_nonorm.npy",
        eeg_count=128,
    )
    val_dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )

    return dl, val_dl
