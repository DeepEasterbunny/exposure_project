import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compute_static_background(device, data_loader = None, model = None):

    static_background = torch.zeros(device=device, size = (1, 128,128))
    total_images = 0
    
    for data in data_loader:
        # If a model is provided, load the synthetic image, and apply the model
        # Else just compute the static background of experimental images
        if model == None:
            _, im, _, _ = data
        else:
            im, _, _, _ = data
            im = im.to(device)
            im = model(im)
        
        im = im.to(device)
        static_background += im.sum(dim=0)
        total_images += im.size(0)

        vflipped = TF.vflip(im)
        static_background += vflipped.sum(dim=0)
        total_images += vflipped.size(0)

        hflipped = TF.hflip(im)
        static_background += hflipped.sum(dim=0)
        total_images += hflipped.size(0)

        vhflipped = TF.hflip(vflipped)
        static_background += vhflipped.sum(dim=0)
        total_images += vhflipped.size(0)
    static_background = static_background.requires_grad_(True)
    # print(static_background)
    return  static_background / total_images

def visualize_images(netG, test_loader, device, resize_transform, epoch, cfg):
    netG.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(5, 3, figsize=(10, 13))
        for i, data in enumerate(test_loader, 0):
            if i >= 5:
                break
            input, target, _, _ = data
            input = input.to(device)
            target = target.to(device)
            output = netG(input)

            # Resize images to 75 x 100
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
        plt.savefig(os.path.join(cfg['saving']['figure_dir'], f'epoch_{epoch}_generated_images_no_kp.png'))
        plt.close(fig)

def visualize_static_background(netG, test_loader, device, experimental_static_background, epoch,cfg):
    generated_static_background_test = compute_static_background(device=device, data_loader=test_loader, model=netG).detach().cpu()
    experimental_static_background = experimental_static_background.detach().cpu()
    
    back_resize_transform = transforms.Resize((75, 100))
    experimental_static_background = back_resize_transform(experimental_static_background[None,:,:]).numpy().squeeze(0)
    generated_static_background_test = back_resize_transform(generated_static_background_test).numpy().squeeze(0)

    # Create a figure to compare the static backgrounds
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(experimental_static_background, cmap='gray')
    axes[0].set_title('Experimental Static Background')
    axes[0].axis('off')

    axes[1].imshow(generated_static_background_test, cmap='gray')
    axes[1].set_title('Generated Static Background (Test Data)')
    axes[1].axis('off')

    plt.suptitle(f'Static Background Comparison at Epoch {epoch}')
    plt.savefig(os.path.join(cfg['saving']['figure_dir'], f'epoch_{epoch}_static_background_comparison.png'))
    plt.close(fig)