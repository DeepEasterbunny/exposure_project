from pathlib import Path
import subprocess

import typer
import os
import hydra
import torch

app = typer.Typer()


@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg):
    print(cfg.foil)
    emtkd(cfg)
    run_emmcfoil()
 
def emtkd(cfg):
    print(cfg.EMTKD)


def run_emmcfoil():
    command = [
        "/zhome/31/8/154954/denoising_in_TKD/EMsoftBuild/Release/Bin/EMMCfoil",
        "/work3/s203768/EMSoftData/simulations/EMMCfoil.nml"
    ]
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

if __name__ == "__main__":
    main()