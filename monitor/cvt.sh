#!/bin/bash

#files=`/usr/bin/ls -1`
files='increments_rdas_conus.py'
for  file in $files
do
echo $file

sed 's/rdas/rdasConus/' $file > $file.new
mv $file.new $file

sed 's/fv3lamda/rdas/' $file > $file.new
mv $file.new $file

sed 's/fv3lam/rrfs/' $file > $file.new
mv $file.new $file

done
