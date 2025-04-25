import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class KikuchiDataset(Dataset):
    def __init__(self, fake, real, detector_values, rots, transform = None):
        """
        Args:
            data (numpy array or torch.Tensor): Shape (n_rots, x, y)
            rots (numpy array or torch.Tensor): Shape (n_rots, 4)
        """
        
        # Ensure fake and real are tensors
        if isinstance(fake, torch.Tensor):
            self.fake = fake.clone().detach()
        else:
            self.fake = torch.tensor(fake, dtype=torch.float32)

        if isinstance(real, torch.Tensor):
            self.real = real.clone().detach()
        else:
            self.real = torch.tensor(real, dtype=torch.float32)
        self.rots = rots.clone().detach()
        self.detector_values = {
            k: torch.tensor(v, dtype=torch.float32) if isinstance(v, (list, tuple, torch.Tensor)) else v
            for k, v in detector_values.items()
        }
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        fake = self.fake[idx]  # Shape (x, y)
        real = self.real[idx]  # Shape (x, y)
        rot = self.rots[idx]

        if self.transform:
            real = self.transform(real[None,:,:])
            fake = self.transform(fake[None,:,:])

        return fake, real, rot, self.detector_values
    