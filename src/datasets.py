# import os
# import numpy as np
# import torch
# from typing import Tuple
# from termcolor import cprint


# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data") -> None:
#         super().__init__()
        
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
        
#         self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, i):
#         if hasattr(self, "y"):
#             return self.X[i], self.y[i], self.subject_idxs[i]
#         else:
#             return self.X[i], self.subject_idxs[i]
        
#     @property
#     def num_channels(self) -> int:
#         return self.X.shape[1]
    
#     @property
#     def seq_len(self) -> int:
#         return self.X.shape[2]

import os
import torch
import torch.utils.data
from .utils import wavelet_transform
from .utils import cmor_transform

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", wavelet: str = 'cmor', level: int = 50) -> None:
        super().__init__()
        
        assert split in ["test", "val", "train"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        # self.X = torch.load(os.path.join(data_dir, f"{split}_X_preprocessed.pt")).float()
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()

        wavelet_file = os.path.join(data_dir, f"{split}_X_wave_{wavelet}{level}.pt")

        if os.path.exists(wavelet_file):
            self.X = torch.load(wavelet_file).float()
        else:
            # Load preprocessed data
            # Apply wavelet transform
            # self.X = wavelet_transform(self.X, wavelet, level)
            self.X = cmor_transform(self.X)
            # Save wavelet transformed data
            torch.save(self.X, wavelet_file)

        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

