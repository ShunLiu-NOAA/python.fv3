#!/bin/bash -l
set +x
. /usrx/local/prod/lmod/lmod/init/sh
set -x

module load impi/18.0.1
module load lsf/10.1
module load HPSS/5.0.2.5

#module use /gpfs/dell3/usrx/local/dev/emc_rocoto/modulefiles/
#module load ruby/2.5.1 rocoto/1.2.4

#module load python/3.6.3
#module use -a /u/Benjamin.Blake/modulefiles
#module load python3/test

bsub < /gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/python.fv3/update_rrfscloud.sh
