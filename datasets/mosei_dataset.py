import h5py
import torch
from torch.utils.data import Dataset

class MOSEIDataset(Dataset):
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, "r")
        self.ids = list(self.h5.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        g = self.h5[sid]

        return {
            "text": torch.tensor(g["text"][:]),
            "audio": torch.tensor(g["audio"][:]),
            "vision": torch.tensor(g["vision"][:]),
            "label": torch.tensor(g["label"][:])
        }
