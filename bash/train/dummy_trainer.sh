#!/bin/sh
#BSUB -q gpuv100
#BSUB -J testjob
#BSUB -n 4
#BSUB -W 0:10
#BSUB -R "rusage[mem=128MB]"
###BSUB -B
###BSUB -N
#BSUB -o hpc_out/training/%J_dummy_without_kp.out
#BSUB -e hpc_out/training/%J_dummy_without_kp.err

nvidia-smi

module load cuda/11.8

source ../tkd/bin/activate

python3 src/noise_gen/dummy.py