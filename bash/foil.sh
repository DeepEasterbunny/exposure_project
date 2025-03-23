#!/bin/sh
#BSUB -q gpuv100
#BSUB -J foil
#BSUB -n 4
# BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:15
#BSUB -R "rusage[mem=1GB]"
#BSUB -u s203768@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o hpc_out/gpu_%J.out
#BSUB -e hpc_out/gpu_%J.err

# Load the cuda module
module load cuda/11.8

#/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
/zhome/31/8/154954/denoising_in_TKD/EMsoftBuild/Release/Bin/EMMCfoil nml/simulations/EMMCfoil.nml
bsub < bash/master.sh