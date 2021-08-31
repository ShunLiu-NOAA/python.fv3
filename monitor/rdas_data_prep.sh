
CYCrun=12
COMOUT=/gpfs/dell1/ptmp/Shun.Liu/rdasConus/rdas.20210831/$CYCrun
tmmark=tm06
datadir=/gpfs/dell1/ptmp/Shun.Liu/stmp/tmpnwprd/testdomain_rrfs_conus_3km/2021083106/anal_gsi_spinup

cd $datadir
cat fit_p1 fit_w1 fit_t1 fit_q1 fit_pw1 fit_rad1 fit_rw1 > $COMOUT/rrfs.t${CYCrun}z.fits.${tmmark}
cat fort.208 fort.210 fort.211 fort.212 fort.213 fort.220 > $COMOUT/rrfs.t${CYCrun}z.fits2.${tmmark}
