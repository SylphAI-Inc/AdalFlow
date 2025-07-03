# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = AdalFlow
SOURCEDIR     = source
BUILDDIR      = build
APIDOCOUTDIR  = $(SOURCEDIR)/apis
PYTHON        := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
POETRY        = poetry

# Put it first so that "make" without argument is like "make help".
help:
	@if command -v sphinx-binary > /dev/null 2>&1; then \
	    $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) ; \
	else \
	    echo "sphinx-binary is not installed. Please grab the latest version of Sphinx (via pip)." ; \
	fi

print_prereqs:
	@for tool in $(SPHINXBUILD) $(POETRY) ; do \
	     if command -v $${tool} > /dev/null 2>&1; then \
	        $${tool} --version ; \
	     else \
	        echo "$${tool} is not installed. Please grab the latest version via pip." ; \
	     fi \
	done

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@sphinx-apidoc -o $(APIDOCOUTDIR)/core ../adalflow/adalflow/core --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/components ../adalflow/adalflow/components --separate --force
#--templatedir=$(SOURCEDIR)/_templates
	@sphinx-apidoc -o $(APIDOCOUTDIR)/eval ../adalflow/adalflow/eval --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/optim ../adalflow/adalflow/optim --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/datasets ../adalflow/adalflow/datasets --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/utils ../adalflow/adalflow/utils --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/tracing ../adalflow/adalflow/tracing --separate --force

update_files:
	@echo "Inserting reference labels into RST files."
	@$(PYTHON) $(SOURCEDIR)/insert_labels.py
	@echo "Removing unnecessary strings for better formatting"
	@$(PYTHON) $(SOURCEDIR)/remove_string.py
	@echo "Removing duplicated files"
	@$(PYTHON) $(SOURCEDIR)/remove_files.py
	@echo "Renaming and updating file"
	@$(PYTHON) $(SOURCEDIR)/change_api_file_name.py
	@echo "Add autosummary of functions and classes"
	@$(PYTHON) $(SOURCEDIR)/insert_autosummary.py

html: apidoc  update_files
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Convenience target to build docs with design documents
build-with-design: apidoc html
	@echo "Documentation built with design documents included!"

# Quick build command
quick:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean and rebuild everything
rebuild: clean build-with-design
