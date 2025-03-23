from pathlib import Path
import subprocess
from omegaconf import OmegaConf
import typer
import numpy as np
import random


def main(cfg_path:str = 'configs/config.yaml'):

    cfg = OmegaConf.load(cfg_path)
    xtal_path = cfg['paths']['xtal']

    xtal_file_names = get_xtals(xtal_path)
    num_master_patterns = cfg['experiments']['n_master_patterns']
    tot = len(xtal_file_names)
    path = create_foil_script(cfg)

    print(f"Found {tot} .xtal files in folder")
    print(f"Creating {num_master_patterns} master patterns per crystal")
    num_job = 1
    tot_jobs = tot * num_master_patterns
    for xtal_file in xtal_file_names:
        for j in range(num_master_patterns):
            foil_name, master_name = create_nmls(cfg, xtal_file)
            print(f"Adding batch job {num_job} of {tot_jobs} to .sh")
            add_to_foil_script(path, foil_name, master_name)
            num_job += 1
    
    print("Created job script")

def create_foil_script(cfg):
    path = Path(cfg['paths']['bash'] + 'send_foil.sh')
    path.touch(exist_ok=True)
    with open(path, 'w') as f:
        f.write('#!/bin/sh\n#BSUB -q hpc\n#BSUB -J foil_script\n#BSUB -n 4\n#BSUB -W 0:15\n#BSUB -R "rusage[mem=512MB]"\n#BSUB -o hpc_out/foil_sender_%J.out\n#BSUB -e hpc_out/foil_sender_%J.err\n')
    return path


def add_to_foil_script(path, foil_name, master_name):
    with open(path, 'a') as f:
        f.write(f'\nbsub -env "all,NML_FILE_FOIL={foil_name},NML_FILE_MASTER={master_name}" < bash/foil_adaptive.sh')


def run_emsoft(cfg, program, file):
    program_path = cfg['paths']['emsoft_bin'] + program
    command = [
        program_path,  file
    ]
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
    
def get_xtals(xtal_path:str):
    xtal_files = [file.name for file in Path(xtal_path).glob("*.xtal")]
    if len(xtal_files) == 0:
        print(f'No .xtal files found in {xtal_path}')
    return xtal_files

def generate_random_config(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    random_config = {}

    for key, value in cfg_dict.items():
        if isinstance(value, list) and len(value) == 2:
            random_config[key] = random.uniform(value[0], value[1])
        else:
            random_config[key] = value

    return random_config

def create_nmls(cfg, xtal_name):
    element = xtal_name.split('.')[0]
    
    foilcfg = generate_random_config(cfg.foil)
    sig = int(np.round(foilcfg['sig'], decimals = 1)*10)
    thickness = int(np.round(foilcfg['thickness'], decimals = 1)*10)
    out_file = Path(f'{element}-master-30kV-sig-{sig}-thickness-{thickness}.h5')
    out_name = 'simulations' / out_file

    foil_name = Path('nml/' + str(out_file).split('.')[0] + '-foil.nml')
    foil_name.touch(exist_ok=True)

    print("Writting nml file for EMMCfoil")
    with open(foil_name, 'w') as f:
        f.write(' &MCCLfoildata\r\n')
        f.write(f" xtalname = '{xtal_name}',\r\n" )
        f.write(f" sig = {np.round(foilcfg['sig'],decimals = 1)},\r\n")
        f.write(" omega = 0.0,\r\n")
        f.write(" numsx = 501,\r\n")
        f.write(" num_el = 10,\r\n")
        f.write(" platid = 1,\r\n")
        f.write(" devid = 1,\r\n")
        f.write(" globalworkgrpsz = 150,\r\n")
        f.write(" totnum_el = 2000000000,\r\n")
        f.write(" multiplier = 1,\r\n")
        f.write(" EkeV = 30.D0,\r\n")
        f.write(f" Ehistmin = {foilcfg['Ehistmin']}.D0,\r\n")
        f.write(" Ebinsize = 1.0D0,\r\n")
        f.write(f" depthmax = {np.round(foilcfg['thickness'], decimals = 1)}D0,\r\n") 
        f.write(" depthstep = 1.0D0,\r\n")
        f.write(f" thickness = {np.round(foilcfg['thickness'], decimals = 1)}D0,\r\n")
        f.write(f" dataname = '{out_name}'\r\n")
        f.write(" /\n")


    print("Writing nml file for EMTKDmaster")
    master_name = Path('nml/' + str(out_file).split('.')[0] + '-master.nml')
    master_name.touch(exist_ok=True)

    master_cfg = generate_random_config(cfg.master)
    with open(master_name, 'r+') as f:
        f.write(" &TKDmastervars\n")
        f.write(f" dmin = {master_cfg['dmin']},\n")
        f.write(f" npx = {master_cfg['npx']},\n")
        f.write(f" energyfile = '{out_name}',\n")
        f.write(f" nthreads = {master_cfg['nthreads']},\n")
        f.write("combinesites = .FALSE.\n")
        f.write(" restart = .FALSE.,\n")
        f.write(" uniform = .FALSE.,\n")
        f.write(" /\n")

    # Check if betheparameters exists
    filename_b = Path('BetheParameters.nml')
    filename_b = 'nml' / filename_b
    if not filename_b.exists():
        print("Creating BetheParameters.nml")
        filename_b.touch()
        with open(filename_b, 'r+') as f:
            f.write(" &Bethelist\n c1 = 4.0, \n c2 = 8.0,\n c3 = 50.0,\n sgdbdiff = 1.00\n /\n")
    else:
        print("BetheParameters.nml already exists, skipping")
    return foil_name, master_name

if __name__ == "__main__":
    typer.run(main)