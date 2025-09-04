# LocAgent LaTeX Documentation

This directory contains the LaTeX source files for the LocAgent paper.

## Setup

We use **tectonic** as our LaTeX engine. Tectonic is a modern, self-contained TeX/LaTeX engine that automatically downloads packages as needed.

### Prerequisites

Make sure you're in the conda `codegen` environment:
```bash
conda activate codegen
```

Tectonic is already installed in this environment.

## Building the Paper

There are several ways to build the paper:

### Option 1: Using Make
```bash
make                 # Build the PDF
make view           # Build and open the PDF
make clean          # Remove temporary files
make distclean      # Remove all generated files including PDF
make watch          # Watch for changes and rebuild automatically
```

### Option 2: Using the build script
```bash
./build.sh          # Build the PDF
./build.sh view     # Build and open the PDF
```

### Option 3: Direct tectonic command
```bash
tectonic acl_latex.tex
```

## Output

The built PDF will be created as `acl_latex.pdf` in the current directory.

## Files

- `acl_latex.tex` - Main LaTeX document
- `acl.sty` - ACL conference style file
- `acl_natbib.bst` - Bibliography style
- `custom.bib` - Bibliography entries
- `Makefile` - Build automation
- `build.sh` - Alternative build script

## Known Issues

1. **Character warnings**: You may see warnings about em dashes (—) and en dashes (–) not being represented in the font. These can be fixed by either:
   - Replacing them with `---` (em dash) and `--` (en dash) in the LaTeX source
   - Using a different font that supports these characters

2. **Underfull hbox warnings**: These are common in LaTeX and usually indicate lines that are too loosely spaced. They can often be ignored unless they affect the visual appearance.

## Tips for Editing

1. The paper uses the ACL style, which requires specific formatting
2. The method section has been added and covers all LocAgent features
3. Citations use natbib format: `\citep{}` for parenthetical, `\citet{}` for textual
4. When adding new sections, maintain consistent style with existing content

## Troubleshooting

If you encounter any issues:

1. Make sure you're in the `codegen` conda environment
2. Try running `make clean` and rebuilding
3. Check that all required files (`.tex`, `.bib`, `.sty`) are present
4. For tectonic-specific issues, it will automatically download missing packages on first run
