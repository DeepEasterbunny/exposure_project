#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ebsd
#BSUB -n 4
# BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:10
#BSUB -R "rusage[mem=512MB]"
#BSUB -u s203768@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o hpc_out/opencl_%J.out
#BSUB -e hpc_out/opencl_%J.err

if [ -z "$NML_FILE_OPENCL" ]; then
  echo "Error: NML_FILE_OPENCL not set"
  exit 1
fi

if [ -z "$NML_FILE_MASTER" ]; then
  echo "Error: NML_FILE_MASTER not set"
  exit 1
fi

module load cuda

/zhome/31/8/154954/denoising_in_TKD/EMsoftBuild/Release/Bin/EMMCOpenCL $NML_FILE_OPENCL
bsub -env "all,NML_FILE_MASTER=$NML_FILE_MASTER" < bash/master_ebsd_adaptive.sh
