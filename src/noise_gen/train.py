import os
from omegaconf import OmegaConf
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

from models import Generator, Discriminator
from utils import initialize_weights, compute_static_background, visualize_images, visualize_static_background, plot_losses, normalized_cross_correlation
from dataset import KikuchiDataset

import typer

FIND_UNIQUE = False
USE_EXISTING_SPLIT = False
REFINE_PATH_G = '/work3/s203768/EMSoftData/checkpoints/flips/30_net_G.pth'
REFINE_PATH_D = '/work3/s203768/EMSoftData/checkpoints/flips/30_net_D.pth'
REFINE = False
starting_epoch = 0

resize_transform = transforms.Resize((128,128))
back_resize_transform = transforms.Resize((75, 100))

# 1st experiment: 63 epochs with 95% of data used for training
# 5gb of memory used
# 8 hours

def train_network(cfg_path:str = 'configs/config_ebsd.yaml', cfg:str = 'configs/config_train.yaml'):
    # Set device
    print("Setting device")
    cfg =  OmegaConf.load(cfg)
    if torch.cuda.is_available():
        print("Running on CUDA")
    else:
        print("Running on CPU")
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    print("Loading networks")
    netG = Generator(cfg['input_nc'], cfg['output_nc'], cfg['ngf']).to(device)
    #print(netG.device)
    netD = Discriminator(cfg['input_nc'], cfg['output_nc'], cfg['ndf']).to(device)

    if REFINE:
        print("Loading networks from checkpoints")
        print(REFINE_PATH_G)
        print(REFINE_PATH_D)
        netG.load_state_dict(torch.load(REFINE_PATH_G, weights_only = True))
        netD.load_state_dict(torch.load(REFINE_PATH_D, weights_only = True))
    else:
        print("Setting weights in new networks")
        netG.apply(initialize_weights)
        netD.apply(initialize_weights)

    # Loss functions
    criterion = nn.BCELoss().to(device)
    criterionAE = nn.L1Loss().to(device)
    criterionStaticBacground = nn.L1Loss().to(device)

    # Optimizers
    cfg_opt = cfg['optimizer']
    optimizerG = optim.Adam(netG.parameters(), lr=cfg_opt['lr'], betas=(cfg_opt['beta1'], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg_opt['lr'], betas=(cfg_opt['beta1'], 0.999))

    transform = transforms.Compose([
        transforms.Resize((128,128))
    ])
    print("Loading data")

    if USE_EXISTING_SPLIT and os.path.exists('train.pt') and os.path.exists('test.pt'):
        print("Loading existing train and test splits...")
        data_dict = torch.load(cfg['data_path'], weights_only=False)
        train_dataset = data_dict['train']
        test_dataset = data_dict['test']
    else:
        data_dict = torch.load(cfg['data_path'], weights_only=False)

        dataset = KikuchiDataset(**data_dict,  transform = transform)  
        if FIND_UNIQUE:
            print("Reducing dataset to only contain unique rotations. To turn this feature off set FIND_UNIQUE = False.")
            print(f"Number of datapoints before {len(dataset)}")

            decimal_places = 1 
            rounded_rots = torch.round(dataset.rots * (10 ** decimal_places)) / (10 ** decimal_places)
            unique_rotations = torch.unique(rounded_rots, dim=0)

            unique_indices = []
            for unique_rot in unique_rotations:
                matches = (rounded_rots == unique_rot).all(dim=1)
                match_idx = matches.nonzero(as_tuple=True)[0][0].item()
                unique_indices.append(match_idx)
                
            unique_indices = torch.tensor(unique_indices, dtype=torch.long)

            dataset.fake = dataset.fake[unique_indices]
            dataset.real = dataset.real[unique_indices]
            dataset.rots = dataset.rots[unique_indices]
            print(f"Number of datapoints after {len(dataset)}")

        train_size = int(cfg['train_split'] * len(dataset))
        print(f"Training on {train_size} images")
        test_size = len(dataset) - train_size
        print(f'Testing on {test_size} images')
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        torch.save(test_dataset, cfg['test_data_path'])
        torch.save(train_dataset, cfg['train_data_path'])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batchSize'], num_workers=cfg['nThreads'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batchSize'], num_workers=cfg['nThreads'], shuffle=True)

    experimental_static_background = compute_static_background(device = device, data_loader=train_loader, model = None)
    static_background_train = torch.empty_like(experimental_static_background).clone()

    # Training loop
    print_freq = cfg['saving']['print_freq']
    epochs = cfg['optimizer']['epochs']
    save_epoch_freq = cfg['saving']['save_epoch_freq']

    G_losses_train = []
    L1_losses_train = []
    sb_losses_train = []
    ncc_losses_train = []
    ncc_bg_losses_train = []
    G_losses_test = []
    L1_losses_test = []
    sb_losses_test = []
    ncc_losses_test = []
    ncc_bg_losses_test = []

    for epoch in range(starting_epoch , epochs):
        l1_loss_train = 0
        errG_loss_train = 0
        sb_loss_train = 0
        ncc_loss_train = 0
        ncc_bg_loss_train = 0
        netG.train()
        netD.train()
        for i, data in enumerate(train_loader, 0):
            real_A, real_B, rotation , _ = data

            real_B = real_B.clone()  
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            rotation = rotation.to(device)
            
            # Add random flips
            if random.random() < 0.5:
                real_A = TF.vflip(real_A)
                real_B = TF.vflip(real_B)
            if random.random() < 0.5:
                real_A = TF.hflip(real_A)
                real_B = TF.hflip(real_B)

            # Update Discriminator
            # Discriminator should believe our simulated image and the target experimental image are "the same"
            netD.zero_grad()
            output = netD(torch.cat((real_A, real_B), dim = 1))
            label = torch.full(output.size(), 1, dtype=torch.float, device=device)
            errD_real = criterion(output, label)
            errD_real.backward()
            # print("First discriminator")

            # Create fake experimental image with our generator
            # Discriminator should believe our simulated image and the target experimental image are NOT "the same"
            fake_B = netG(real_A)
            output = netD(torch.cat((real_A, fake_B.detach()), dim = 1) )
            label.fill_(0)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
            # print("Second discriminator")

            # Update Generator
            # Take the image we just created, feed it to the updated Discriminator, our generator is rewarded if the discriminator is wrong
            # Also add and L1 loss, to encourage looking like the experimental image
            
            netG.zero_grad()
            output = netD(torch.cat((real_A, fake_B.detach()), dim = 1))
            label.fill_(1)

            
            errG = criterion(output, label)
            errL1 = criterionAE(fake_B, real_B) * cfg['loss']['lambda_L1']


            # Cross correlation - higer = better
            # (quaternion) anglewidth, doesn't take into accout symmetry
            # generated_static_background = compute_static_background(device=device, data_loader=train_loader, model=netG)

            generated_static_background = compute_static_background(device=device, batch = fake_B)

            loss_static_background = criterionStaticBacground(experimental_static_background, generated_static_background) 

            ncc_loss = normalized_cross_correlation(real_A, fake_B, static_background_train)
            ncc_bg_loss = normalized_cross_correlation(real_A, fake_B)
            


            errG_total = errG + errL1 + loss_static_background * cfg['loss']['sb'] + (-ncc_bg_loss)*3

            errG_loss_train += errG.item()
            l1_loss_train += errL1.item()
            sb_loss_train += loss_static_background.item()
            ncc_loss_train += ncc_loss.item()
            ncc_bg_loss_train += ncc_bg_loss.item()

            errG_total.backward(retain_graph=True)
            # print(f"Done with batch {i+1}/{len(train_loader)} in epoch {epoch+1}/{epochs}")
            optimizerG.step()
    

        G_losses_train.append(errG_loss_train / len(train_loader))
        L1_losses_train.append(l1_loss_train / len(train_loader))
        sb_losses_train.append((sb_loss_train / len(train_loader)))
        ncc_losses_train.append(ncc_loss_train / len(train_loader))
        ncc_bg_losses_train.append(ncc_bg_loss_train / len(train_loader))

        netG.eval()
        netD.eval()

        l1_loss_test = 0
        errG_loss_test = 0
        sb_loss_test = 0
        ncc_loss_test = 0
        ncc_bg_loss_test = 0
        for i, data in enumerate(test_loader, 0):
            real_A, real_B, rotation , _ = data

            real_B = real_B.clone()  
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            rotation = rotation.to(device)
            
            # Add random flips
            if random.random() < 0.5:
                real_A = TF.vflip(real_A)
                real_B = TF.vflip(real_B)
            if random.random() < 0.5:
                real_A = TF.hflip(real_A)
                real_B = TF.hflip(real_B)

            fake_B = netG(real_A)

            output = netD(torch.cat((real_A, fake_B.detach()), dim = 1))

            label = torch.full(output.size(), 1, dtype=torch.float, device=device)

            errG = criterion(output, label)
            errL1 = criterionAE(fake_B, real_B) * cfg['loss']['lambda_L1']
            generated_static_background = compute_static_background(device=device, batch=fake_B)

            loss_static_background = criterionStaticBacground(experimental_static_background, generated_static_background) 

            ncc_loss = normalized_cross_correlation(real_A, fake_B, static_background_train)
            ncc_bg_loss = normalized_cross_correlation(real_A, fake_B)


            errG_total = errG + errL1 + loss_static_background* cfg['loss']['sb'] + (1 - ncc_loss)

            errG_loss_test += errG.item()
            l1_loss_test += errL1.item()
            sb_loss_test += loss_static_background.item()
            ncc_loss_test += ncc_loss.item()
            ncc_bg_loss_test += ncc_bg_loss.item()

        G_losses_test.append(errG_loss_test / len(test_loader))
        L1_losses_test.append(l1_loss_test / len(test_loader))
        sb_losses_test.append((sb_loss_test / len(test_loader)))
        ncc_losses_test.append((ncc_loss_test / len(test_loader) ))
        ncc_bg_losses_test.append((ncc_bg_loss_test / len(test_loader)))

        plot_losses(G_losses_train, L1_losses_train, sb_losses_train, G_losses_test, L1_losses_test, sb_losses_test, ncc_losses_train, ncc_bg_losses_train, ncc_losses_test, ncc_bg_losses_test , cfg)


        if (epoch + 1) % save_epoch_freq == 0:
            print(f'End of epoch {epoch+1}/{epochs}')
        
        # if (epoch + 1) % 3 == 0 and epoch > 15:
        #         print(f"End of epoch {epoch}")
        #         print("Updating static background")
        #         sb = compute_static_background(device=device, data_loader=train_loader, model=netG)
        #         static_background_train = sb.clone()

    # Save final models
    torch.save(netG.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], 'final_net_G.pth'))
    torch.save(netD.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], 'final_net_D.pth'))


if __name__ == '__main__':
    torch.manual_seed(42)
    train_network()
    # typer.run(train_network)