#!/bin/sh
#BSUB -q gpuv100
#BSUB -J training
#BSUB -n 4
#BSUB -W 8:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -u s203768@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o hpc_out/training/%J.out
#BSUB -e hpc_out/training/%J.err

module load cuda

source ../tkd/bin/activate

python3 src/noise_gen/train.py