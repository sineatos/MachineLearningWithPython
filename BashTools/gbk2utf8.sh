#!/bin/bash
  
if [ "$#" != "2" ]; then
  
  echo "Usage: `basename $0` dir filter"
  
  exit
  
fi
  
dir=$1
  
filter=$2
  
echo $1
  
for file in `find $dir -name "$2"`; do
  
  echo "$file"
  
  iconv -f gbk -t utf8 -o $file $file
  
done
