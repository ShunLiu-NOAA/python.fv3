#!/bin/ksh --login

#USER=Benjamin.Blake
#PDY=20190517
#cyc=00

cd /gpfs/dell3/stmp/${USER}/damonitor/${PDY}/${cyc}

# Retrieve main.php to update cycle dates
scp wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/main6.php .

DATE=$(sed -n "191p" main6.php | cut -c 15-24)
DATEm1=$(sed -n "191p" main6.php | cut -c 28-37)
DATEm2=$(sed -n "191p" main6.php | cut -c 41-50)
DATEm3=$(sed -n "191p" main6.php | cut -c 54-63)
DATEm4=$(sed -n "191p" main6.php | cut -c 67-76)
DATEm5=$(sed -n "191p" main6.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '191s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main6.php > tmpfile ; mv tmpfile main6.php

scp main6.php wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs


scp wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/main7.php .

DATE=$(sed -n "234p" main7.php | cut -c 15-24)
DATEm1=$(sed -n "234p" main7.php | cut -c 28-37)
DATEm2=$(sed -n "234p" main7.php | cut -c 41-50)
DATEm3=$(sed -n "234p" main7.php | cut -c 54-63)
DATEm4=$(sed -n "234p" main7.php | cut -c 67-76)
DATEm5=$(sed -n "234p" main7.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '234s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main7.php > tmpfile ; mv tmpfile main7.php

scp main7.php wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs


# Copy namelist and model configure files to emcrzdm - need to automate conversion to html here
#scp /gpfs/hps3/emc/meso/save/wx22h/rrfssar/regional_workflow/parm/input_sar_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22h/rrfssar/regional_workflow/parm/model_configure_sar.html
#scp /gpfs/hps3/emc/meso/save/wx22h/rrfssarx/regional_workflow/parm/input_sarx_conus.html
#scp /gpfs/hps3/emc/meso/save/wx22h/rrfssarx/regional_workflow/parm/model_configure_sarx.html

# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/damonitor/images/*.png"

# move cycm4 images to cycm5 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/damonitor/images/lamda*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm5/damonitor/images/"

# move cycm3 images to cycm4 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/damonitor/images/lamda*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm4/damonitor/images/"

# move cycm2 images to cycm3 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/damonitor/images/lamda*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm3/damonitor/images/"

# move cycm1 images to cycm2 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/damonitor/images/lamda*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm2/damonitor/images/"

# move cyc images to cycm1 directory
ssh wx22hl@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/damonitor/images/lamda*.png /home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cycm1/damonitor/images/"


# Copy images from WCOSS to emcrzdm
rsync -t lamda*.png wx22hl@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/mmb/wx22hl/rrfs/cyc/damonitor/images/


exit
