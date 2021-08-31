#!/bin/bash

#files=`cat list`
files=`/usr/bin/ls -1 launch*`
for  file in $files
do
sed 's/.bashrc/bin\'/loadp.sh'/g' $file > $file.new
mv $file.new $file
chmod +x $file
done
