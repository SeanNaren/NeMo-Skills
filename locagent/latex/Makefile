# Makefile for building LaTeX documents with tectonic

# Main document name (without .tex extension)
MAIN = acl_latex

# Default target
all: $(MAIN).pdf

# Build the PDF using tectonic
$(MAIN).pdf: $(MAIN).tex custom.bib
	@echo "Building $(MAIN).pdf with tectonic..."
	tectonic $(MAIN).tex

# Alternative: build with bibtex support explicitly
full: $(MAIN).tex custom.bib
	@echo "Building $(MAIN).pdf with full bibliography support..."
	tectonic -X compile $(MAIN).tex

# Clean temporary files (tectonic handles most of this automatically)
clean:
	@echo "Cleaning temporary files..."
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f *.synctex.gz

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(MAIN).pdf

# Watch for changes and rebuild automatically
watch:
	@echo "Watching for changes (press Ctrl+C to stop)..."
	@while true; do \
		inotifywait -e modify $(MAIN).tex custom.bib 2>/dev/null; \
		make; \
	done

# View the PDF (using default PDF viewer)
view: $(MAIN).pdf
	@echo "Opening $(MAIN).pdf..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf; \
	elif command -v open > /dev/null; then \
		open $(MAIN).pdf; \
	else \
		echo "No PDF viewer found. Please open $(MAIN).pdf manually."; \
	fi

# Help message
help:
	@echo "Available targets:"
	@echo "  make         - Build the PDF"
	@echo "  make full    - Build with explicit bibliography support"
	@echo "  make clean   - Remove temporary files"
	@echo "  make distclean - Remove all generated files including PDF"
	@echo "  make watch   - Watch for changes and rebuild automatically"
	@echo "  make view    - Build and open the PDF"
	@echo "  make help    - Show this help message"

.PHONY: all full clean distclean watch view help
