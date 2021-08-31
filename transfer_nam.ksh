#!/bin/ksh --login

#USER=Benjamin.Blake
#CDATE=2019051700
#cyc=00

cd /gpfs/dell3/stmp/${USER}/fv3nam/${CDATE}/${cyc}

# Retrieve main2.php to update cycle dates
scp wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/main2.php .

DATE=$(sed -n "286p" main2.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main2.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main2.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main2.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main2.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main2.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main2.php > tmpfile ; mv tmpfile main2.php

scp main2.php wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs

# Copy namelist and model configure files to emcrzdm - need to automate conversion to html here
#echo `date -u` > input_lam_conus.html
#cat /gpfs/dell6/emc/modeling/noscrub/wx22hl/fv3lam/regional_workflow/parm/input_lam_conus.html > input_lam_conus.html
#scp /gpfs/dell6/emc/modeling/noscrub/wx22hl/fv3lam/regional_workflow/parm/input_lam_conus.html wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/namelists
#echo `date -u` > model_configure_lam.html
#cat /gpfs/dell6/emc/modeling/noscrub/wx22hl/fv3lam/regional_workflow/parm/model_configure_lam.html > model_configure_lam.html
#scp /gpfs/dell6/emc/modeling/noscrub/wx22hl/fv3lam/regional_workflow/parm/model_configure_lam.html wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/namelists
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/input_sarx_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/model_configure_sarx.html


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/nam/images/*.png"

# move cycm4 images to cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/nam/images/"

# move cycm3 images to cycm4 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/nam/images/"

# move cycm2 images to cycm3 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/nam/images/"

# move cycm1 images to cycm2 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/nam/images/"

# move cyc images to cycm1 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/nam/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *conus*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *BN*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *CE*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *CO*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *LA*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *MA*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *NC*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *NE*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *NW*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *OV*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *SC*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *SE*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *SF*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *SP*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *SW*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/
rsync -t *UM*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/nam/images/


exit
