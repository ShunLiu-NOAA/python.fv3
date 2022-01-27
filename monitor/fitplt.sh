#!/bin/ksh
#set -x

USER=Shun.Liu
#sdate=2022011900

#PDY=20220121
#cyc=00

PDY=$1
cyc=$2

sdate=${PDY}$cyc

echo 'into launch_profiles'
. /usrx/local/prod/lmod/lmod/init/sh
source /u/${USER}/bin/loadp.sh

mkdir -p /gpfs/dell3/stmp/${USER}
mkdir -p /gpfs/dell3/stmp/${USER}/damonitor
mkdir -p /gpfs/dell3/stmp/${USER}/damonitor/$PDY
mkdir -p /gpfs/dell3/stmp/${USER}/damonitor/$PDY/$cyc
cd /gpfs/dell3/stmp/${USER}/damonitor/$PDY/$cyc

#plotexe=/gpfs/dell2/emc/modeling/noscrub/${USER}/PyGSI/scripts/plot_gsi_stat_exp_all.py 
plotexe=/gpfs/dell2/emc/modeling/noscrub/${USER}/PyGSI/scripts/plot_gsi_stat_exp_shun.py 


fit1=/gpfs/dell2/emc/modeling/noscrub/Shun.Liu/test/fitobs/fv3lamdax_shun.$PDY/$cyc
#fit1=/gpfs/dell2/emc/modeling/noscrub/Shun.Liu/test/fitobs/fv3lamdax_emc.$PDY/$cyc
fit2=/gpfs/dell2/emc/modeling/noscrub/Shun.Liu/test/fitobs/fv3lamdax_emc.$PDY/$cyc

#tmmark=tm06

cd /gpfs/dell3/stmp/${USER}/damonitor/$PDY/$cyc
#totaltm='tm06 tm05 tm04 tm03 tm02 tm01 tm00'
totaltm='tm00'
for tmmark in $totaltm
do
  python $plotexe -d1 $fit1 -d2 $fit2 -s ${sdate} -e ${sdate} -t $tmmark -f
done

convert lamda_uvtq_RMSE_tm00.png lamda_uvtq_Bias_tm00.png +append 1.png
convert 1.png lamda_uvtq_Count_tm00.png -append fit2obs_${sdate}.png
rm -f 1.png
scpfv3 fit2obs_${sdate}.png




exit
python /gpfs/dell2/emc/modeling/noscrub/${USER}/PyGSI/scripts/plot_gsi_stat_exp_all.py -d1 /gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.${PDY}/${cyc} -d2 /gpfs/dell2/ptmp/emc.campara/fv3lamdax/fv3lamdax.${PDY}/${cyc} -s ${sdate} -e ${sdate} -t tm06 -f

exit
