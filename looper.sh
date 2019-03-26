#!/bin/bash

for ((INDEX=0;INDEX<=14;INDEX++)); do
	python3 dataset_creator.py -i $INDEX
done
