#!/bin/bash

let i=1

if [[ $# -lt 1 ]]
then
	echo Usage: $0 iterations
fi

MAX_ITER=$1

while [[ $i -le $MAX_ITER ]]
do

	echo $i
	python train_classifier.py --model feedforward --epochs 15 --batch_size 291 --batches 49 --directory tower --algo optimal_rrt --scene tower_defense --split 0.95 --truncation $i --step_size 100
	let i+=1

done
