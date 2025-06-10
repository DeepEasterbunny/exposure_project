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
# DICT_PATH = '/work3/s203768/EMSoftData/dicts/Dict_Ni_in_use.h5'

def generate_images(cfg_ebsd:str = 'configs/config_ebsd.yaml', cfg_train:str = 'configs/config_train.yaml'):
    cfg_train =  OmegaConf.load(cfg_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = torch.load(cfg_train['train_data_path'], weights_only=False, map_location=device)
    test_dataset = torch.load(cfg_train['test_data_path'], weights_only=False, map_location = device)


    transform = transforms.Compose([
        transforms.Resize((75,100))
    ])
    # train_dataset = KikuchiDataset(**train_dataset,  transform = transform)
    # test_dataset = KikuchiDataset(**test_dataset,  transform = transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg_train['batchSize'], num_workers=cfg_train['nThreads'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg_train['nThreads'], shuffle=True)

    netG = Generator(cfg_train['input_nc'], cfg_train['output_nc'], cfg_train['ngf'])

    netG.load_state_dict(torch.load(GENERATOR_PATH, weights_only = True, map_location  = device))

    
    fake_patterns = np.zeros(shape = (len(test_loader), 75, 100))
    real_rotations = np.zeros(shape = (len(test_loader), 4))
    real_patterns = np.zeros(shape = (len(test_loader), 75, 100))

    real_static_background = compute_static_background(device, test_loader)
    fake_static_background = compute_static_background(device, test_loader, netG)

    real_static_background = transform(real_static_background).detach().numpy().squeeze()
    fake_static_background = transform(fake_static_background).detach().numpy().squeeze()

    netG.eval()
    for (i,data) in enumerate(test_loader):
        fake, real, rots, _ = data
        generated_image = netG(fake)
        fake_patterns[i] = transform(generated_image).detach().numpy()
        real_patterns[i] = transform(real).detach().numpy()
        real_rotations[i] = rots.detach().numpy()


    return fake_patterns, real_patterns, real_rotations, fake_static_background, real_static_background

        
def create_histograms(fake_patterns, real_patterns, fake_static_background, real_static_background):

    fake_patterns = (fake_patterns - np.min(fake_patterns, axis = (1,2), keepdims=True)) / (fake_patterns.max(axis=(1, 2), keepdims=True) - fake_patterns.min(axis=(1, 2), keepdims=True))

    real_patterns = (real_patterns - np.min(real_patterns, axis = (1,2), keepdims=True)) / (real_patterns.max(axis=(1, 2), keepdims=True) - real_patterns.min(axis=(1, 2), keepdims=True))

    fake_static_background = (fake_static_background - np.min(fake_static_background, keepdims=True)) / (fake_static_background.max(keepdims=True) - fake_static_background.min(keepdims=True))

    real_static_background = (real_static_background - np.min(real_static_background, keepdims=True)) / (real_static_background.max(keepdims=True) - real_static_background.min(keepdims=True))


    plt.figure(figsize=(10, 5))
    plt.hist(real_patterns.flatten(), bins=50, alpha=0.5, label="Real Patterns", color="blue")
    plt.hist(fake_patterns.flatten(), bins=50, alpha=0.5, label="Fake Patterns", color="orange")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Real and Fake Patterns")
    plt.legend()
    plt.savefig("results/histogram_real_fake_patterns.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(real_static_background.flatten(), bins=50, alpha=0.5, label="Real Static Background", color="green")
    plt.hist(fake_static_background.flatten(), bins=50, alpha=0.5, label="Fake Static Background", color="red")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Real and Fake Static Backgrounds")
    plt.legend()
    plt.savefig("results/histogram_real_fake_static_background.png")
    plt.close()

if __name__ == '__main__':

    # Try https://orix.readthedocs.io/en/stable/reference/generated/orix.quaternion.Rotation.angle_with.html#orix.quaternion.Rotation.angle_with
    # Calculate all angles and take the smallest
    # Take into account symmetry
    
    print("Generating fake patterns from test data")
    fake_patterns, real_patterns, real_rotations_yuxan, fake_static_background, real_static_background = generate_images()

    print("Loading dict, and performing dictionary indexing on test data")
    dic = kp.load(DICT_PATH)
    detector = dic.detector
    signal_mask = ~kp.filters.Window("circular", detector.shape).astype(bool)
    fake_signal = kp.signals.EBSD(fake_patterns- fake_static_background, detector = detector )

    real_signal = kp.signals.EBSD(real_patterns- real_static_background, detector = detector )
 
    xmap_fake = fake_signal.dictionary_indexing(dictionary = dic, keep_n = 1, metric = 'ncc', n_per_iteration = 4000, signal_mask = signal_mask)
    print(f"Indexing with Lucas' dictionary on the fake data: {xmap_fake.scores.mean()}")

    print(xmap_fake.rotations)

    rots_fake = dic.xmap.rotations[xmap_fake.simulation_indices[:]].data
    # print(rots_fake)

    xmap_real = real_signal.dictionary_indexing(dictionary = dic, keep_n = 1, metric = 'ncc', n_per_iteration = 4000, signal_mask = signal_mask)
    print(f"Indexing with Lucas' dictionary on the real data: {xmap_real.scores.mean()}")

    real_rots = dic.xmap.rotations[xmap_real.simulation_indices[:]].data

    fig, axes = plt.subplots(1, 2, figsize=(7, 14))
    axes[0].imshow(fake_patterns[0], cmap='gray')
    axes[0].set_title('Fake Pattern')
    axes[0].axis('off')
    axes[1].imshow(real_patterns[0], cmap='gray')
    axes[1].set_title('Real Pattern')
    axes[1].axis('off')
    plt.savefig('results/fake_real_patterns.png')

    sym = oqu.symmetry.Oh  # m-3m symmetry

    ors_fake = []
    ors_mine = []
    ors_yuxan = []
    print("Converting np arrays to orientations")
    for quarternion in rots_fake:
        ors_fake.append(oqu.Orientation(quarternion, sym))

    for quarternion in real_rots:
        ors_mine.append(oqu.Orientation(quarternion.squeeze(), sym))
        
    for quarternion in real_rotations_yuxan:
        ors_yuxan.append(oqu.Orientation(quarternion.squeeze(), sym))

    angs_mine = []
    angs_yuxan = []
    my_dict_yuxan = []
    for i, ori in enumerate(ors_fake):
    
        angs_mine.append(ori.angle_with(ors_mine[i], degrees = True))
        angs_yuxan.append(ori.angle_with(ors_yuxan[i], degrees = True))
        my_dict_yuxan.append(ors_mine[i].angle_with(ors_yuxan[i], degrees = True))

    np.savetxt(fname="results/misorientations_with_my_dict.txt", X = np.array(angs_mine))
    np.savetxt(fname="results/misorientations_with_yuxan_dict.txt", X = np.array(angs_yuxan))
    np.savetxt(fname="results/misorientations_with_yuxan_dict_my_dict.txt", X = np.array(my_dict_yuxan))

    create_histograms(fake_patterns=fake_patterns, real_patterns=real_patterns, real_static_background=real_static_background, fake_static_background=fake_static_background)

    
    

    