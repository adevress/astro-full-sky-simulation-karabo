#!/bin/bash

# Script to convert all Jupyter notebooks in the root directory to Python files
# Usage: ./convert_notebooks.sh

# Find all .ipynb files in the root directory
NOTEBOOKS=$(find . -maxdepth 1 -name "*.ipynb" -type f)

# Check if any notebooks were found
if [ -z "$NOTEBOOKS" ]; then
    echo "No Jupyter notebooks (.ipynb files) found in the root directory."
    exit 0
fi

# Convert each notebook to Python file
echo "Converting Jupyter notebooks to Python files..."
for notebook in $NOTEBOOKS; do
    # Get the base filename without extension
    base_name=$(basename "$notebook" .ipynb)
    
    # Convert to Python file
    echo "Converting $notebook to ${base_name}.py"
    jupyter nbconvert --to script "$notebook"
    
    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        echo "Successfully converted $notebook"
    else
        echo "Failed to convert $notebook"
    fi
done

echo "Conversion complete!"
