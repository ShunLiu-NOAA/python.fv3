  if [ ! -z $MODULESHOME ]; then
      . $MODULESHOME/init/bash
  else
      . /opt/modules/default/init/bash
  fi
  
  module load ips/18.0.1.163
  module load impi/18.0.1
  module load NetCDF/4.5.0
  module load bacio/2.0.2
  module load sfcio/1.0.0
  module load lsf/10.1
  module load nemsio/2.2.3
  module load w3emc/2.3.0
  module load sp/2.0.2
  module load w3nco/2.0.6
  module load impi/18.0.1
  module load bufr/11.2.0
  module load sigio/2.0.1
  module load crtm/2.2.5
  
  module load EnvVars/1.0.2
  module load pm5/1.0
  module load subversion/1.7.16
  module load HPSS/5.0.2.5
  module load mktgs/1.0
  module load rsync/3.1.2
  module load ip/3.0.1
  module load prod_envir/1.0.2
  module load grib_util/1.0.6
  module load prod_util/1.1.0
  module load bufr_util/1.0.1
  
  module use /gpfs/dell3/usrx/local/dev/emc_rocoto/modulefiles/
  module load ruby/2.5.1 rocoto/1.2.4
  
  module use -a /usrx/local/dev/modulefiles
  module load git/2.14.3
  
  module load python/3.6.3
  module use -a /u/Benjamin.Blake/modulefiles
  module load python3/test
  module load wgrib2/2.0.8
  #module load anaconda2/latest
  #export GRIB_DEFINITION_PATH=/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/EXT/grib_api.1.14.4/share/grib_api/definitions
  export GRIB_DEFINITION_PATH=/usrx/local/nceplibs/dev/lib/grib_api/share/grib_api/definitions
  export PYTHONPATH="${PYTHONPATH}:/gpfs/dell2/emc/verification/noscrub/Logan.Dawson/python:/gpfs/dell2/emc/modeling/noscrub/Jacob.Carley/python/lib/python3.6/site-packages:/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/PyGSI"

