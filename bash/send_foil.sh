#!/bin/sh
#BSUB -q hpc
#BSUB -J foil_script
#BSUB -n 4
#BSUB -W 0:15
#BSUB -R "rusage[mem=512MB]"
#BSUB -o hpc_out/foil_sender_%J.out
#BSUB -e hpc_out/foil_sender_%J.err

bsub -env "all,NML_FILE_FOIL=nml/Ni-master-30kV-sig-0-thickness-489-foil.nml,NML_FILE_MASTER=nml/Ni-master-30kV-sig-0-thickness-489-master.nml" < bash/foil_adaptive.sh
bsub -env "all,NML_FILE_FOIL=nml/Ni-master-30kV-sig-0-thickness-434-foil.nml,NML_FILE_MASTER=nml/Ni-master-30kV-sig-0-thickness-434-master.nml" < bash/foil_adaptive.sh