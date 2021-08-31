#!/bin/ksh --login

#USER=Benjamin.Blake
#CDATE=2019051700
#cyc=00

cd /gpfs/dell3/stmp/${USER}/4panel/${CDATE}/${cyc}

# Retrieve main.php to update cycle dates
scp wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/main3.php .

DATE=$(sed -n "286p" main3.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main3.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main3.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main3.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main3.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main3.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main3.php > tmpfile ; mv tmpfile main3.php

scp main3.php wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs

# Copy namelist and model configure files to emcrzdm - need to automate conversion to html here
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssar/regional_workflow/parm/input_sar_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssar/regional_workflow/parm/model_configure_sar.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/input_sarx_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/model_configure_sarx.html

# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/compare/images/*.png"

# move cycm4 images to cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/compare/images/"

# move cycm3 images to cycm4 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/compare/images/"

# move cycm2 images to cycm3 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/compare/images/"

# move cycm1 images to cycm2 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/compare/images/"

# move cyc images to cycm1 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/compare/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *conus*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *BN*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *CE*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *CO*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *LA*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *MA*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *NC*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *NE*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *NW*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *OV*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *SC*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *SE*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *SF*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *SP*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *SW*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/
rsync -t *UM*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/compare/images/


exit
