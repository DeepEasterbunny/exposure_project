import torch
from torch.utils.data import Dataset

class KikuchiDataset(Dataset):
    def __init__(self, fake, real, detector_values, transform = None):
        """
        Args:
            data (numpy array or torch.Tensor): Shape (n_rots, x, y)
            rots (numpy array or torch.Tensor): Shape (n_rots, 4)
        """
        # assert len(data) == len(rots), "Mismatch between data and rotations"
        
        self.fake = torch.tensor(fake, dtype=torch.float32)  # Convert to tensor
        self.real = torch.tensor(real, dtype=torch.float32)  # Convert to tensor
        # self.rots = torch.tensor(rots, dtype=torch.float32)  # Convert to tensor
        self.detector_values = {
            k: torch.tensor(v, dtype=torch.float32) if isinstance(v, (list, tuple, torch.Tensor)) else v
            for k, v in detector_values.items()
        }
        self.transform = transform

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        fake = self.fake[idx]  # Shape (x, y)
        real = self.real[idx]  # Shape (x, y)
        if self.transform:
            real = self.transform(real[None,:,:])
            fake = self.transform(fake[None,:,:])
        # rotation = self.rots[idx]  # (1 x 4)
        return fake, real,  self.detector_values
    