#!/bin/bash

# Script to clean all Jupyter notebooks in the root directory by removing outputs
# Usage: ./clean_notebooks.sh

echo "Cleaning Jupyter notebooks by removing all outputs..."

# Find all .ipynb files in the root directory
NOTEBOOKS=$(find . -maxdepth 1 -name "*.ipynb" -type f)

# Check if any notebooks were found
if [ -z "$NOTEBOOKS" ]; then
    echo "No Jupyter notebooks (.ipynb files) found in the root directory."
    exit 0
fi

# Clean each notebook by removing outputs
for notebook in $NOTEBOOKS; do
    echo "Cleaning $notebook..."
    
    # Use jupyter nbconvert to clear outputs
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$notebook"
    
    # Check if cleaning was successful
    if [ $? -eq 0 ]; then
        echo "Successfully cleaned $notebook"
    else
        echo "Failed to clean $notebook"
    fi
done

echo "Notebook cleaning complete!"
