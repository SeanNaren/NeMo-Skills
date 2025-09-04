#!/bin/bash

# Build script for LaTeX document using tectonic

echo "Building acl_latex.pdf with tectonic..."

# Run tectonic
# The -X compile flag ensures proper bibliography handling
tectonic -X compile acl_latex.tex

if [ $? -eq 0 ]; then
    echo "Build successful! Output: acl_latex.pdf"
    
    # Optionally open the PDF if the argument "view" is passed
    if [ "$1" == "view" ]; then
        if command -v xdg-open > /dev/null; then
            xdg-open acl_latex.pdf
        elif command -v open > /dev/null; then
            open acl_latex.pdf
        else
            echo "PDF viewer not found. Please open acl_latex.pdf manually."
        fi
    fi
else
    echo "Build failed! Check the error messages above."
    exit 1
fi
