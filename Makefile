pdf:
	pdflatex -shell-escape main.tex
	bibtex main.aux
	pdflatex -shell-escape main.tex
	pdflatex -shell-escape main.tex
clean:
	(rm -rf *.ps *.log *.dvi *.aux *.*% *.lof *.lop *.lot *.toc *.idx *.ilg *.ind *.bbl *.blg *.cpt *.tdo)
	rm -rf src/*.pyc
