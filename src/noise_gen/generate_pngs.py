import torch
import matplotlib.pyplot as plt
import os
import glob
from omegaconf import OmegaConf
import typer
import numpy as np

from dataset import KikuchiDataset

def to_euler(quaternion):
    """
    Converts a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees.
    """
    w, x, y, z = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.degrees(np.sign(sinp) * (np.pi / 2))  # Clamp to 90 degrees
    else:
        pitch = np.degrees(np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    
    return roll, pitch, yaw

def get_element_and_thickness(pt_file):
    file_name = os.path.basename(pt_file)
    v2 = file_name.split('.')[0]
    
    return file_name.split('-master')[0], v2.split('thickness-')[1]


def main(cfg_path:str = 'configs/config.yaml', n_sets:int = 4, n_images:int = 4):
    """
    Loads up to `n_sets` .pt files from a folder and creates PNG images.
    Each PNG contains `n_images` columns and up to 3 rows, where each row is from a new .pt file.
    """
    cfg = OmegaConf.load(cfg_path)
    torch_path = cfg['paths']['torch_data']
    output_folder = cfg['paths']['images']
    
    # Find all .pt files in the folder
    pt_files = glob.glob(os.path.join(torch_path, "*.pt"))
    
    if not pt_files:
        print("No .pt files found in the folder.")
        return
    
    # Load datasets from .pt files
    datasets = []
    elements = []
    thicknesses = []
    for pt_file in pt_files:
        e, t = get_element_and_thickness(pt_file=pt_file)
        elements.append(e)
        thicknesses.append(t)
        data_dict = torch.load(pt_file, weights_only=False)  # Load the .pt file
        dataset = KikuchiDataset(**data_dict)  # Create a dataset object
        datasets.append(dataset)

    num_figures = len(datasets) // 3 + (1 if len(datasets) % 3 else 0)
    
    for fig_idx in range(num_figures):
        fig, axes = plt.subplots(3, n_images + 1, figsize=((n_images + 1) * 3, 9))
        fig.suptitle(f"Kikuchi Patterns - Set {fig_idx+1}")
        
        for row in range(3):
            dataset_idx = fig_idx * 3 + row
            if dataset_idx >= len(datasets):
                break  # No more datasets to process
            
            dataset = datasets[dataset_idx]
            element = elements[dataset_idx]
            thickness = thicknesses[dataset_idx]
            
            # Add detector values in the first column
            pattern_center = [dataset.detector_values['pcx'],
                              dataset.detector_values['pcy'],
                              dataset.detector_values['pcz']]
            detector_text = (f"Element: {element}\nThickness: {thickness} Å\n "
                             f"Pattern Center: {pattern_center}\n"
                             f"Sample tilt {dataset.detector_values['sample_tilt']} deg")
            axes[row, 0].text(0.5, 0.5, detector_text, fontsize=10, ha='center', va='center', wrap=True)
            axes[row, 0].axis("off")
            
            for col in range(n_images):
                ax = axes[row, col + 1] if n_images > 1 else axes[row, 1]
                if col >= len(dataset):
                    ax.axis("off")  # Hide empty plots
                    continue
                
                image, rotation, _ = dataset[col]  # Extract image
                ax.imshow(image.numpy(), cmap="gray")
                ax.set_title(f"Euler-angels: {np.round(to_euler(rotation.numpy()), decimals = 1)}", fontsize=8)
                ax.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        output_path = os.path.join(output_folder, f"kikuchi_set_{fig_idx+1}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved {output_path}")

if __name__ == '__main__':
    typer.run(main)