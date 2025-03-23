#!/bin/sh
#BSUB -q hpc
#BSUB -J opencl_script
#BSUB -n 4
#BSUB -W 0:15
#BSUB -R "rusage[mem=512MB]"
#BSUB -o hpc_out/opencl_sender_%J.out
#BSUB -e hpc_out/opencl_sender_%J.err

bsub -env "all,NML_FILE_OPENCL=nml/simulations_ebsd/Fe-master-30kV-sig-0-thickness-300-opencl.nml,NML_FILE_MASTER=nml/simulations_ebsd/Fe-master-30kV-sig-0-thickness-300-master.nml" < bash/opencl_adaptive.sh