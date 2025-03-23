#!/bin/sh
#BSUB -q hpc
#BSUB -J pattern_generator
#BSUB -n 4
#BSUB -W 2:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -u s203768@dtu.dk
###BSUB -B
###BSUB -N
#BSUB -o hpc_out/pattern_generator_%J.out
#BSUB -e hpc_out/pattern_generator_%J.err

source ../tkd/bin/activate

python3 src/noise_gen/tkd_detector.py