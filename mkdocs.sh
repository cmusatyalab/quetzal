#!/bin/bash

# Navigate to the project directory if necessary
# cd /path/to/your/project

# make sure you do chmod +x mkdocs.sh

# Run pdoc to generate HTML documentation for the quetzal package
pdoc --html --force ./quetzal --output-dir ./docs

echo "Documentation has been generated in ./docs directory."