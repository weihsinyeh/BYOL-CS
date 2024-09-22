#!/bin/bash

# TODO - run your inference Python3 code
if [ "$#" -ne 3 ]; then
    epoch "Usage: bash hw1_1.sh <path_to_images_csv> <path_to_images_folder> <output_csv_file>"
    exit 1
output_csv_file

IMAGES_CSV = $1
IMAGES_FOLDER = $2
OUTPUT_CSV = $3

if [! -f "$IMAGES_CSV" ]; then
    echo "Error: Images CSV file does not exist."
    exit 1
fi
if [! -d "$IMAGES_FOLDER" ]; then
    echo "Error: Images folder does not exist."
    exit 1
fi
python evaluate.py "$IMAGES_CSV" "$IMAGES_FOLDER" "$OUTPUT_CSV"

if [ $? -ne 0 ]; then
    echo "Error: Predictin script failed."
    exit 1
fi

# Train backbone model.
# python hw1/main.py