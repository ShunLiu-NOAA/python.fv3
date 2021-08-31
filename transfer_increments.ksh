#!/bin/ksh --login

#USER=Benjamin.Blake
#PDY=20190517
#cyc=00

cd /gpfs/dell3/stmp/${USER}/increments/${CDATE}
rm -f lamda*.png

# Retrieve main.php to update cycle dates
scp wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/main8.php .

DATE=$(sed -n "224p" main8.php | cut -c 15-24)
DATEm1=$(sed -n "224p" main8.php | cut -c 28-37)
DATEm2=$(sed -n "224p" main8.php | cut -c 41-50)
DATEm3=$(sed -n "224p" main8.php | cut -c 54-63)
DATEm4=$(sed -n "224p" main8.php | cut -c 67-76)
DATEm5=$(sed -n "224p" main8.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '224s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main8.php > tmpfile ; mv tmpfile main8.php

scp main8.php wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs


# Copy namelist and model configure files to emcrzdm - need to automate conversion to html here
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssar/regional_workflow/parm/input_sar_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssar/regional_workflow/parm/model_configure_sar.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/input_sarx_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22hl/rrfssarx/regional_workflow/parm/model_configure_sarx.html

# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/increments/images/*.png"

# move cycm4 images to cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/increments/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/increments/images/"

# move cycm3 images to cycm4 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/increments/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/increments/images/"

# move cycm2 images to cycm3 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/increments/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/increments/images/"

# move cycm1 images to cycm2 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/increments/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/increments/images/"

# move cyc images to cycm1 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/increments/images/*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/increments/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/increments/images/


exit
