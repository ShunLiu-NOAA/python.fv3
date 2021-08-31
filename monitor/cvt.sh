#!/bin/bash

files=`/usr/bin/ls -1`
for  file in $files
do
#sed 's/emc.campara/wx22hl/g' $file > $file.new
#mv $file.new $file
echo $file
done
