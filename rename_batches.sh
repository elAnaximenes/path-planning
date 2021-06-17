#!/bin/bash

if [ $# -lt 2 ]
then
	echo "usage: $0 folder_name start_index"
	exit 2
fi

FOLDER=$1
START_IDX=$2
let IDX=$START_IDX

for FNAME in `ls ./$FOLDER/*`
do
	echo $FNAME 
	#NEW_NAME="${FNAME:0:31}$IDX"
	echo $IDX
	let IDX++
done


