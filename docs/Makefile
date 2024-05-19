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
	@echo "Running sphinx-apidoc for core"
	@sphinx-apidoc -o $(APIDOCOUTDIR)/core ../core --separate --force --module-first
	@echo "Running sphinx-apidoc for components"
	@sphinx-apidoc -o $(APIDOCOUTDIR)/components ../components --separate --force --module-first



html: apidoc
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)