import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
plt.style.use(["seaborn-deep", "seaborn-whitegrid"])

import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce

# The digit classes to use, these need to be in order because
# we are using one-hot representation
classes = np.arange(2)

def one_hot(labels):
    y = torch.eye(len(classes)) 
    return y[labels]

# Define the train and test sets
dset_train = MNIST("./", train=True, download=True, transform=ToTensor(), target_transform=one_hot)
dset_test  = MNIST("./", train=False, transform=ToTensor(), target_transform=one_hot)

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)


batch_size = 64
# The loaders perform the actual work
train_loader = DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.train_labels), pin_memory=cuda)
test_loader  = DataLoader(dset_test, batch_size=batch_size, 
                          sampler=stratified_sampler(dset_test.test_labels), pin_memory=cuda)

from torch import nn

latent_dim = 100

# The generator takes random `latent` noise and
# turns it into an MNIST image.
generator = nn.Sequential(
    # nn.ConvTranspose2d can be seen as the inverse operation
    # of Conv2d, where after convolution we arrive at an
    # upscaled image.
    nn.ConvTranspose2d(latent_dim, 256, kernel_size=3, stride=2),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
    nn.Sigmoid() # Image intensities are in [0, 1]
).to(device)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# The discriminator takes an image (real or fake)
# and decides whether it is generated or not.
discriminator = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=4, stride=2),
    nn.LeakyReLU(0.2),
    nn.Conv2d(64, 128, kernel_size=4, stride=2),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2),
    nn.Conv2d(128, 256, kernel_size=4, stride=2),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2),
    Flatten(),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(device)

loss = nn.BCELoss()
print("Using device:", device)

generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))

from torch.autograd import Variable
import os

tmp_img = "tmp_gan_out.png"
discriminator_loss, generator_loss = [], []

num_epochs = 50
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    
    for x, _ in train_loader:
        batch_size = x.size(0)
        # True data is given label 1, while fake data is given label 0
        true_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        
        discriminator.zero_grad()
        generator.zero_grad()
        
        # Step 1. Send real data through discriminator
        #         and backpropagate its errors.
        x_true = Variable(x).to(device)        
        output = discriminator(x_true)
        
        error_true = loss(output, true_label)
        error_true.backward()
        
        # Step 2. Generate fake data G(z), where z ~ N(0, 1)
        #         is a latent code.
        z = torch.randn(batch_size, latent_dim, 1, 1)
        z = Variable(z, requires_grad=False).to(device)
        
        x_fake = generator(z)
            
        # Step 3. Send fake data through discriminator
        #         propagate error and update D weights.
        # --------------------------------------------
        # Note: detach() is used to avoid compounding generator gradients
        output = discriminator(x_fake.detach()) 
        
        error_fake = loss(output, fake_label)
        error_fake.backward()
        discriminator_optim.step()
        
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = discriminator(x_fake)
        
        error_generator = loss(output, true_label)
        error_generator.backward()
        generator_optim.step()
        
        batch_d_loss.append((error_true/(error_true + error_fake)).item())
        batch_g_loss.append(error_generator.item())

    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    
    # -- Plotting --
    f, axarr = plt.subplots(1, 2, figsize=(18, 7))

    # Loss
    ax = axarr[0]
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.plot(np.arange(epoch+1), discriminator_loss)
    ax.plot(np.arange(epoch+1), generator_loss, linestyle="--")
    ax.legend(['Discriminator', 'Generator'])
    
    # Latent space samples
    ax = axarr[1]
    ax.set_title('Samples from generator')
    ax.axis('off')

    rows, columns = 8, 8
    
    # Generate data
    with torch.no_grad():
        z = torch.randn(rows*columns, latent_dim, 1, 1)
        z = Variable(z, requires_grad=False).to(device)
        x_fake = generator(z)
    
    canvas = np.zeros((28*rows, columns*28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_fake.data[idx]
    ax.imshow(canvas, cmap='gray')
    
    plt.savefig(tmp_img)
    plt.close(f)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)