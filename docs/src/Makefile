# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build
SRCDIR        = .

# See https://www.sphinx-doc.org/en/master/man/sphinx-build.html?highlight=--keep-going#cmdoption-sphinx-build-W
WARNING_TO_ERROR = -W --keep-going

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean html html-noplot dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest show

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  help          to view this help"
	@echo "  html          to make standalone HTML files"
	@echo "  html-noplot   to make standalone HTML files, skip gallery"
	@echo "  html-nitpicky to make standalone HTML files and  use sphinx nitpicky"	
	@echo "  html-noapi    to make standalone HTML files, skip the API"
	@echo "  html-quick    to make standalone HTML files, skip the gallery and API"
	@echo "  clean         to remove all built files"
	@echo "  doctest       to run all doctests embedded in the documentation (if enabled)"
	@echo "  linkcheck     to check all external links for integrity"
	@echo "  show          to open the built documentation in the default browser"
	@echo "  livehtml      to auto build the docs when any file changes are detected."
	@echo "                You need to install sphinx-autobuild first:"
	@echo "                    conda install -c conda-forge sphinx-autobuild"

html:
	$(SPHINXBUILD) $(WARNING_TO_ERROR) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html"

html-nitpicky:
	$(SPHINXBUILD) -n -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML (no api docs) pages are in $(BUILDDIR)/html"

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML (no gallery docs) pages are in $(BUILDDIR)/html"

html-noapi:
	export SKIP_API=1; $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML (no api docs) pages are in $(BUILDDIR)/html"

html-quick:
	export SKIP_API=1; $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML (no gallery or api docs) pages are in $(BUILDDIR)/html"

clean:
	-rm -rf $(BUILDDIR)
	-rm -rf $(SRCDIR)/generated

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the "
	@echo "results in $(BUILDDIR)/doctest/output.txt."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output "
	@echo "or in $(BUILDDIR)/linkcheck/output.txt."

show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(shell pwd)/$(BUILDDIR)/html/index.html')"

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) --ignore generated $(O)