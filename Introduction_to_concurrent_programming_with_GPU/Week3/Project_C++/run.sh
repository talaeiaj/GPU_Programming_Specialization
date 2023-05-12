#!/usr/bin/env bash
make clean build

NUM_RUNS=$1
USERNAME=$2
NUM_THREADS=$3
PART_ID=$4

for i in $(seq 1 ${NUM_RUNS})
do
	make run ARGS="$NUM_THREADS $USERNAME $PART_ID"
done