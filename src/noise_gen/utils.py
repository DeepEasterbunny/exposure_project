import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
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


def compute_static_background_im(im, static_background, total_images):
    static_background += im.sum(dim=0)
    total_images += im.size(0)

    # vflipped = TF.vflip(im)
    # static_background += vflipped.sum(dim=0)
    # total_images += vflipped.size(0)

    # hflipped = TF.hflip(im)
    # static_background += hflipped.sum(dim=0)
    # total_images += hflipped.size(0)

    # vhflipped = TF.hflip(vflipped)
    # static_background += vhflipped.sum(dim=0)
    # total_images += vhflipped.size(0)

    return static_background, total_images



def compute_static_background(device, data_loader = None, model = None, batch = None):

    static_background = torch.zeros(device=device, size = (1, 128,128))
    total_images = 0

    assert not ((data_loader != None) and (batch != None)), "Please provide only one of a Dataloader or batch"
    assert not ((data_loader == None) and (batch == None)), "Please provide either a dataloader or a batch"

    if batch is not None:
        static_background, total_images = compute_static_background_im(batch, static_background, total_images)
    
    if data_loader is not None:
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
            static_background, total_images = compute_static_background_im(im, static_background, total_images)

    static_background = static_background.requires_grad_(True)
    
    return  static_background / total_images

def visualize_images(netG, test_loader, device, resize_transform, epoch, cfg):
    netG.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(5, 3, figsize=(10, 13))
        i = 0
        for data in test_loader:
    
            input, target, _, _ = data
            input = input.to(device)
            target = target.to(device)
            output = netG(input)

            # Resize images to 75 x 100
            input = resize_transform(input.cpu())
            output = resize_transform(output.cpu())
            target= resize_transform(target.cpu())

            for _ in range(input.size(0)):
                if i >= 5:
                    break
                axes[i, 0].imshow(input[i,0,:,:], cmap='gray')
                axes[i, 0].set_title('Input')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(output[i,0,:,:], cmap='gray')
                axes[i, 1].set_title('Output')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(target[i,0,:,:], cmap = 'gray')
                axes[i, 2].set_title('Target')
                axes[i, 2].axis('off')
                i += 1
        

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


def plot_losses(G_train_loss, L1_train_loss, sb_train_loss, G_test_loss, L1_test_loss, sb_test_loss, ncc_loss_train, ncc_bg_loss_train, ncc_loss_test, ncc_bg_loss_test , cfg):
    epoch = len(G_train_loss)
    epochs = range(epoch)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].plot(epochs, G_train_loss)
    axes[0,0].plot(epochs, G_test_loss)
    axes[0,0].set_title('Generator loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend(['Train', 'Test'])

    axes[0,0].plot(epochs, G_train_loss)
    axes[0,0].plot(epochs, G_test_loss)
    axes[0,0].set_title('Generator loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend(['Train', 'Test'])

    axes[0,1].plot(epochs, L1_train_loss)
    axes[0,1].plot(epochs, L1_test_loss)
    axes[0,1].set_title('L1 loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend(['Train', 'Test'])

    axes[0,2].plot(epochs, sb_train_loss)
    axes[0,2].plot(epochs, sb_test_loss)
    axes[0,2].set_title('Static Background loss')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Loss')
    axes[0,2].legend(['Train', 'Test'])

    # print(epochs)

    axes[1,0].plot(epochs, ncc_loss_train)
    axes[1,0].plot(epochs, ncc_loss_test)
    axes[1,0].set_title('NCC loss')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].legend(['Train', 'Test'])

    axes[1,1].plot(epochs, ncc_bg_loss_train)
    axes[1,1].plot(epochs, ncc_bg_loss_test)
    axes[1,1].set_title('NCC Background dominated loss')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Loss')
    axes[1,1].legend(['Train', 'Test'])

    plt.suptitle(f'Loss functions at epoch: {epoch}')
    plt.savefig(os.path.join(cfg['saving']['figure_dir'], f'loss_funs.png'))
    plt.close(fig)

def normalized_cross_correlation(image1, image2, static_background = None):
    if static_background is not None:
        image2 = image2 - static_background

    image1 = image1.squeeze(1)
    image2 = image2.squeeze(1)

    image1 = (image1 - image1.amin(dim=(1,2), keepdim=True)) / (image1.amax(dim=(1,2), keepdim=True) - image1.amin(dim=(1,2), keepdim=True))

    image2 = (image2 - image2.amin(dim=(1,2), keepdim=True)) / (image2.amax(dim=(1,2), keepdim=True) - image2.amin(dim=(1,2), keepdim=True))

    image1 = image1 - image1.mean(dim=(1,2), keepdim=True)
    image2 = image2 - image2.mean(dim=(1,2), keepdim=True)

    numerator = (image1 * image2).sum(dim=(1,2))
    denominator = torch.sqrt((image1 ** 2).sum(dim=(1,2)) * (image2 ** 2).sum(dim=(1,2)))

    result = torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)

    return result.mean()  
