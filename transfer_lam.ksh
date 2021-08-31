#!/bin/ksh --login

#USER=Benjamin.Blake
#CDATE=2019051700
#cyc=00

cd /gpfs/dell3/stmp/${USER}/oconus/${CDATE}/${cyc}

# Retrieve main.php to update cycle dates
scp wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/main4.php .

DATE=$(sed -n "286p" main4.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main4.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main4.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main4.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main4.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main4.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main4.php > tmpfile ; mv tmpfile main4.php

scp main4.php wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs


scp wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/main5.php .

DATE=$(sed -n "242p" main5.php | cut -c 15-24)
DATEm1=$(sed -n "242p" main5.php | cut -c 28-37)
DATEm2=$(sed -n "242p" main5.php | cut -c 41-50)
DATEm3=$(sed -n "242p" main5.php | cut -c 54-63)
DATEm4=$(sed -n "242p" main5.php | cut -c 67-76)
DATEm5=$(sed -n "242p" main5.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '242s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main5.php > tmpfile ; mv tmpfile main5.php

scp main5.php wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs


# Copy namelist and model configure files to emcrzdm - need to automate conversion to html here
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssar/regional_workflow/parm/input_sar_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssar/regional_workflow/parm/model_configure_sar.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/input_sarx_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/model_configure_sarx.html

# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/oconus/images/*.png"

# move cycm4 images to cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/monitor/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/monitor/images/"

# move cycm3 images to cycm4 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/monitor/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/monitor/images/"

# move cycm2 images to cycm3 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/monitor/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/monitor/images/"

# move cycm1 images to cycm2 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/monitor/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/monitor/images/"

# move cyc images to cycm1 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/*f0*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/*f1*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/*f2*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/*f3*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/*f4*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/oconus/images/"
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/monitor/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/monitor/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *namerica*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/
rsync -t *ak*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/
rsync -t *hi*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/
rsync -t *pr*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/oconus/images/

# Copy any images generated from monitor scripts to emcrzdm
cd /gpfs/dell3/stmp/${USER}/monitor/${CDATE}
rsync -t *.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/monitor/images/


exit
