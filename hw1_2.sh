#!/bin/bash

# TODO - run your inference Python3 code

# bash hw1_2.sh $1 $2
# $1: testing images directory with images named 'xxxx_sat.jpg’ (e.g., hw1_hiddendata/p2_data/test/)
# $2: output images directory (e.g., output_p2/pred_dir/)

# Example : Usage
# bash hw1_2.sh ./hw1_data/p2_data/validation ./Pb2_evaluation

if [ "$#" -ne 2 ]; then
    echo "Usage: bash hw1_2.sh <path_to_test_images_folder> <path_to_output_folder>"
    exit 1
fi

TEST_IMAGES_FOLDER=$1
OUTPUT_FOLDER=$2

if [ ! -d "$TEST_IMAGES_FOLDER" ]; then
    echo "Error: Test images folder does not exist."
    exit 1
fi

python3 evaluation_hw1_2.py --data_test_dir "$TEST_IMAGES_FOLDER" --output_dir "$OUTPUT_FOLDER"