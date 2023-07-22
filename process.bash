#!/bin/bash

# Set the input and output directories
input_dir="img"
output_dir="img"

# Loop over all PDF files in the input directory
for pdf_file in "$input_dir"/*.pdf; do
    # Get the base filename without the extension
    base_filename=$(basename "$pdf_file" .pdf)
    # Set the output filename with the PNG extension
    output_file="$output_dir/$base_filename.png"
    # Convert the PDF to PNG using convert
    convert -density 300 "$pdf_file" -quality 90 "$output_file"
done