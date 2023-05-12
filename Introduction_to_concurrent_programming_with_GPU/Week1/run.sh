make clean build

NUM_RUNS=$1
USERNAME=$2

for i in $(seq 1 $NUM_RUNS)
do
	make run ARGS="$USERNAME"
done