#!/bin/sh
#BSUB -q hpc
#BSUB -J masterpatterns
#BSUB -n 4
##### BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -u s203768@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o hpc_out/master_%J.out
#BSUB -e hpc_out/master_%J.err

# Load the cuda module
# module load cuda/11.8

/zhome/31/8/154954/denoising_in_TKD/EMsoftBuild/Release/Bin/EMTKDmaster nml/simulations/EMTKDmaster.nml
bsub < bash/pattern.sh
