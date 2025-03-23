import os
from omegaconf import OmegaConf
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt



from models import Generator, Discriminator
from utils import initialize_weights
from dataset import KikuchiDataset

import typer

def train_network(cfg_path:str = 'configs/config_ebsd.yaml', cfg:str = 'configs/config_train.yaml'):
    # Set device
    print("Setting device")
    cfg_ebsd = OmegaConf.load(cfg_path)
    cfg =  OmegaConf.load(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    print("Loading networks")
    netG = Generator(cfg['input_nc'], cfg['output_nc'], cfg['ngf']).to(device)
    netD = Discriminator(cfg['input_nc'], cfg['output_nc'], cfg['ndf']).to(device)

    # Initialize weights
    print("Setting weights")
    netG.apply(initialize_weights)
    netD.apply(initialize_weights)

    # Loss functions
    criterion = nn.BCELoss().to(device)
    criterionAE = nn.L1Loss().to(device)

    # Optimizers
    cfg_opt = cfg['optimizer']
    optimizerG = optim.Adam(netG.parameters(), lr=cfg_opt['lr'], betas=(cfg_opt['beta1'], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg_opt['lr'], betas=(cfg_opt['beta1'], 0.999))

    # Data loader
    # transform = transforms.Compose([
    #     transforms.Resize(cfg['image-size']['loadSize']),
    #     transforms.RandomCrop(cfg['image-size']['fineSize']),
    #     transforms.RandomHorizontalFlip() if cfg['augment']['flip'] else transforms.Lambda(lambda x: x),
    #     transforms.ToTensor()
    # ])
    transform = transforms.Compose([
        transforms.Resize((128,128))
    ])
    print("Loading data")
    data_dict = torch.load(cfg['data_path'], weights_only=False)
    dataset = KikuchiDataset(**data_dict,  transform = transform)  

    train_size = int(cfg['train_split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batchSize'], num_workers=cfg['nThreads'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batchSize'], num_workers=cfg['nThreads'], shuffle=False)

    # Training loop
    print_freq = cfg['saving']['print_freq']
    epochs = cfg['optimizer']['epochs']
    save_epoch_freq = cfg['saving']['save_epoch_freq']

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            real_A, real_B, _ = data
            real_B = real_B.clone()  

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Update Discriminator
            netD.zero_grad()
            output = netD(torch.cat((real_A, real_B), dim = 1))
            label = torch.full(output.size(), 1, dtype=torch.float, device=device)
            errD_real = criterion(output, label)
            errD_real.backward()

            fake_B = netG(real_A)
            output = netD(torch.cat((real_A, fake_B.detach()), dim = 1) )
            label.fill_(0)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            output = netD(torch.cat((real_A, fake_B.detach()), dim = 1))
            label.fill_(1)
            errG = criterion(output, label)
            errL1 = criterionAE(fake_B, real_B) * cfg['loss']['lambda_L1']
            errG_total = errG + errL1
            errG_total.backward()
            optimizerG.step()

            if i % print_freq == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{i}/{len(train_loader)}] '
                      f'Loss_D: {errD_real.item() + errD_fake.item()} Loss_G: {errG.item()} Loss_L1: {errL1.item()}')

        if epoch % save_epoch_freq == 0:
            torch.save(netG.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], f'{epoch}_net_G.pth'))
            torch.save(netD.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], f'{epoch}_net_D.pth'))
        
            netG.eval()
            with torch.no_grad():
                fig, axes = plt.subplots(5, 3, figsize=(10, 16))
                for i, data in enumerate(test_loader, 0):
                    if i >= 5:
                        break
                    input, target, _ = data
                    input = input.to(device)
                    target = target.to(device)
                    output = netG(input)

                    # Resize images to 75 x 100
                    resize_transform = transforms.Resize((75, 100))
                    input = resize_transform(input.cpu())
                    output = resize_transform(output.cpu())
                    target= resize_transform(target.cpu())


                    axes[i, 0].imshow(input[0,0,:,:], cmap='gray')
                    axes[i, 0].set_title('Input')
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(output[0,0,:,:], cmap='gray')
                    axes[i, 1].set_title('Output')
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(target[0,0,:,:], cmap = 'gray')
                    axes[i, 2].set_title('Target')
                    axes[i, 2].axis('off')

                plt.suptitle(f'Epoch {epoch}')
                plt.savefig(os.path.join(cfg['saving']['figure_dir'], f'epoch_{epoch}_generated_images.png'))
                plt.close(fig)

        print(f'End of epoch {epoch}/{epochs}')

    # Save final models
    torch.save(netG.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], 'final_net_G.pth'))
    torch.save(netD.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], 'final_net_D.pth'))


if __name__ == '__main__':
    torch.manual_seed(42)
    train_network()
    # typer.run(train_network)