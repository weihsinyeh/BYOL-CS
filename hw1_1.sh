#!/bin/bash

# TODO - run your inference Python3 code

# $1: path to the images csv file (e.g., hw1_hiddendata/p1_data/office/test.csv)
# $2: path to the folder containing images (e.g., hw1_hiddendata/p1_data/office/test/)
# $3: path of output .csv file (predicted labels) (e.g., output_p1/test_pred.csv)

# Example : Office-Home dataset Usage
# bash hw1_1.sh ./hw1_data/p1_data/office/val.csv ./hw1_data/p1_data/office/val PB1_output.csv

if [ "$#" -ne 3 ]; then
    epoch "Usage: bash hw1_1.sh <path_to_images_csv> <path_to_images_folder> <output_csv_file>"
    exit 1
fi

IMAGES_CSV=$1
IMAGES_FOLDER=$2
OUTPUT_CSV=$3

if [ ! -f "$IMAGES_CSV" ]; then
    echo "Error: Images CSV file does not exist."
    exit 1
fi
if [ ! -d "$IMAGES_FOLDER" ]; then
    echo "Error: Images folder does not exist."
    exit 1
fi
python3 evaluation_hw1_1.py --test_csv_file "$IMAGES_CSV" --finetune_test_dir "$IMAGES_FOLDER" --output_csv_file "$OUTPUT_CSV"