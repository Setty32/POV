PDFLATEX=pdflatex --shell-escape --halt-on-error --synctex=1
BIBTEX=bibtex

RM=rm -rf

.PHONY: clean clean-all nothing

nothing: # Nothing rule

%.pdf: %.tex
	$(PDFLATEX) "$*"

%.bbl: %.aux
	$(BIBTEX) "$*"

clean:
	$(RM) $(addprefix *.,aux bbl blg log toc nav out snm)

clean-all: clean
	$(RM) $(addprefix *.,pdf)
