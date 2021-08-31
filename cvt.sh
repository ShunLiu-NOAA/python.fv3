#!/bin/bash

files=`cat list`
for  file in $files
do
sed 's/emc.campara/wx22hl/g' $file > $file.new
mv $file.new $file
done
