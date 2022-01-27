#!/bin/ksh
#set -x

USER=Shun.Liu
#sdate=2022011900
PDY=20220125
cyc=00
sdate=${PDY}$cyc

echo 'into getfit.sh'

fitdir=/gpfs/dell2/ptmp/Shun.Liu/fv3lamdax/fv3lamdax.$PDY/$cyc
rdasNA=/gpfs/dell2/emc/modeling/noscrub/Shun.Liu/test/fitobs/fv3lamdax_shun.$PDY/$cyc
mkdir -p $rdasNA
cp $fitdir/*fits.* $rdasNA 
cd $rdasNA
tm='tm06 tm05 tm04 tm03 tm02 tm01 tm00'
for itm in $tm
do
  sed -e 's/asm 900     /asm 900 0000/' fv3lam.t${cyc}z.fits.$itm > $itm
  mv $itm fv3lam.t${cyc}z.fits.$itm
done

fitdirp=/gpfs/dell2/ptmp/emc.campara/fv3lamdax/fv3lamdax.$PDY/$cyc
rdasNAt=/gpfs/dell2/emc/modeling/noscrub/Shun.Liu/test/fitobs/fv3lamdax_emc.$PDY/$cyc
mkdir -p $rdasNAt
cp $fitdirp/*fits.* $rdasNAt 
cd $rdasNAt
tm='tm06 tm05 tm04 tm03 tm02 tm01 tm00'
for itm in $tm
do
  sed -e 's/asm 900     /asm 900 0000/' fv3lam.t${cyc}z.fits.$itm > $itm
  mv $itm fv3lam.t${cyc}z.fits.$itm
done

exit
