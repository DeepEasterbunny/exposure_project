#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Train_gpu
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s203768@dtu.dk
###BSUB -B
###BSUB -N
#BSUB -o hpc_out/training/%J_gpu.out
#BSUB -e hpc_out/training/%J_gpu.err


# 
# export CUDA_LAUNCH_BLOCKING=1
# Load the cuda module
module load cuda/11.8
# nvidia-smi

source ../tkd/bin/activate

python3 src/noise_gen/train.py