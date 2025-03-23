#!/bin/sh
#BSUB -q hpc
#BSUB -J masterpatterns
#BSUB -n 4
#BSUB -W 0:30
#BSUB -R "rusage[mem=1GB]"
#BSUB -u s203768@dtu.dk
###BSUB -B
###BSUB -N
#BSUB -o hpc_out/master_%J.out
#BSUB -e hpc_out/master_%J.err


if [ -z "$NML_FILE_MASTER" ]; then
  echo "Error: nml file not set"
  exit 1
fi

/zhome/31/8/154954/denoising_in_TKD/EMsoftBuild/Release/Bin/EMEBSDmaster $NML_FILE_MASTER
