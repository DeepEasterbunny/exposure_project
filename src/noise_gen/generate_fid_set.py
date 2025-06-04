import numpy as np
import torch
from torch.utils.data import Dataset
from dataset import KikuchiDataset
from omegaconf import OmegaConf
from models import Generator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kikuchipy as kp
from utils import compute_static_background
import orix.quaternion as oqu
import matplotlib.pyplot as plt

GENERATOR_PATH = '/work3/s203768/EMSoftData/checkpoints/flips/final_net_G.pth'
DICT_PATH = '/work3/s203768/EMSoftData/dicts/Ni-master-30kV-sig-0-thickness-300.h5'

def generate_images(cfg_ebsd:str = 'configs/config_ebsd.yaml', cfg_train:str = 'configs/config_train.yaml'):
    cfg_train =  OmegaConf.load(cfg_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = torch.load(cfg_train['test_data_path'], weights_only=False, map_location = device)


    transform = transforms.Compose([
        transforms.Resize((75,100))
    ])

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg_train['nThreads'], shuffle=True)

    netG = Generator(cfg_train['input_nc'], cfg_train['output_nc'], cfg_train['ngf'])

    netG.load_state_dict(torch.load(GENERATOR_PATH, weights_only = True, map_location  = device))

    netG.eval()
    for (i,data) in enumerate(test_loader):
        fake, real, _, _ = data
        generated_image = netG(fake)
        
        generated_image = transform(generated_image)
        real = transform(real)

        real_img = real.squeeze().cpu().numpy()
        gen_img = generated_image.squeeze().detach().cpu().numpy()

        plt.imsave('/work3/s203768/EMSoftData/figs/fid/real/' + f"{i}.png", real_img, cmap='gray')
        plt.imsave('/work3/s203768/EMSoftData/figs/fid/generated/' + f"{i}.png", gen_img, cmap='gray')

if __name__ == '__main__':
    generate_images()