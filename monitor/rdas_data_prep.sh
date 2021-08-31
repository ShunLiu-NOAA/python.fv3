
CYCrun=12
CDATE=20210831
COMOUT=/gpfs/dell1/ptmp/Shun.Liu/rdasConus/rdas.$CDATE/$CYCrun
prodcyc=$CDATE$CYCrun

totaltm='00 01 02 03 04 05 06'

for tm in $totaltm
do
  tmmark=tm$tm
  let tm1=CYCrun-tm
  cycspinup=$(printf "%02d" $tm1)
  thiscyc=$CDATE$cycspinup
  echo $tmmark, $cycspinup, $thiscyc
 

  if [ $cycspinup == 12 ]; then
    datadir=/gpfs/dell1/ptmp/Shun.Liu/stmp/tmpnwprd/testdomain_rrfs_conus_3km/$thiscyc/anal_gsi
  else
    datadir=/gpfs/dell1/ptmp/Shun.Liu/stmp/tmpnwprd/testdomain_rrfs_conus_3km/$thiscyc/anal_gsi_spinup
  fi
  
  cd $datadir
  cat fit_p1 fit_w1 fit_t1 fit_q1 fit_pw1 fit_rad1 fit_rw1 > $COMOUT/rrfs.t${CYCrun}z.fits.${tmmark}
  cat fort.208 fort.210 fort.211 fort.212 fort.213 fort.220 > $COMOUT/rrfs.t${CYCrun}z.fits2.${tmmark}
  #echo $datadir
done


exit
