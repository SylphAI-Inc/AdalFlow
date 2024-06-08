# Minimal makefile for Sphinx documentation
#


# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = LightRAG
SOURCEDIR     = source# the source of output and conf.py
BUILDDIR      = build
APIDOCOUTDIR  = $(SOURCEDIR)/apis

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@sphinx-apidoc -o $(APIDOCOUTDIR)/core ../lightrag/core --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/components ../lightrag/components --separate --force --templatedir=$(SOURCEDIR)/_templates	
	@sphinx-apidoc -o $(APIDOCOUTDIR)/eval ../lightrag/eval --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/utils ../lightrag/utils --separate --force
	@sphinx-apidoc -o $(APIDOCOUTDIR)/tracing ../lightrag/tracing --separate --force
	@echo "Inserting reference labels into RST files."
	@python $(SOURCEDIR)/insert_labels.py
	@echo "Removing unnecessary strings for better formatting"
	@python $(SOURCEDIR)/remove_string.py
	@echo "Removing duplicated files"
	@python $(SOURCEDIR)/remove_files.py



html: apidoc
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)