#!/bin/sh
#BSUB -q gpuv100
#BSUB -J foil
#BSUB -n 4
#BSUB -gpu"num=1:mode=exclusive_process"
#BSUB -W 0:15
#BSUB -R "rusage[mem=1GB]"
#BSUB -u s203768@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o hpc_out/foil_%J.out
#BSUB -e hpc_out/foil_%J.err

if [ -z "$NML_FILE_FOIL" ]; then
  echo "Error: NML_FILE_FOIL not set"
  exit 1
fi

if [ -z "$NML_FILE_MASTER" ]; then
  echo "Error: NML_FILE_MASTER not set"
  exit 1
fi

module load cuda

/zhome/31/8/154954/denoising_in_TKD/EMsoftBuild/Release/Bin/EMMCfoil $NML_FILE_FOIL
bsub -env "all,NML_FILE_MASTER=$NML_FILE_MASTER" < bash/master_adaptive.sh
