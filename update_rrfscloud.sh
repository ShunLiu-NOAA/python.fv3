#!/bin/sh
#BSUB -oo /gpfs/dell3/ptmp/Benjamin.Blake/logfiles2/update_rrfscloud.log
#BSUB -eo /gpfs/dell3/ptmp/Benjamin.Blake/logfiles2/update_rrfscloud.log
#BSUB -J update_rrfscloud
#BSUB -W 00:05
#BSUB -P RRFS-T2O
#BSUB -q "dev_transfer"
#BSUB -R affinity[core]
#BSUB -R rusage[mem=1000]

USER=Benjamin.Blake
#CDATE=2021052700
#PDY=20210527
PDY=`cut -c 7-14 /gpfs/dell1/nco/ops/com/date/t00z`
cyc=00
CDATE=${PDY}${cyc}

mkdir -p /gpfs/dell3/stmp/${USER}/rrfscloud
cd /gpfs/dell3/stmp/${USER}/rrfscloud
rm -f *.php

# Retrieve main.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main9.php .

# Line 226 replace PDY
DATEm1=$(sed -n "226p" main9.php | cut -c 10-17)
echo $DATEm1
sed '226s/var cyc="'${DATEm1}'"/var cyc="'${PDY}'"/' main9.php > tmpfile ; mv tmpfile main9.php

# Line 244 append CDATE
CDATEm1=$(sed -n "244p" main9.php | cut -c 15-24)
echo $CDATEm1
sed '244s/"'${CDATEm1}'"/"'${CDATE}'", &/' main9.php > tmpfile ; mv tmpfile main9.php

# Line 245 append PDY
sed '245s/"'${DATEm1}'"/"'${PDY}'", &/' main9.php > tmpfile ; mv tmpfile main9.php


scp main9.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


exit
