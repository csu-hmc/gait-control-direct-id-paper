Introduction
============

This is the source repository for the paper:

   Moore, J.K. and van den Bogert, A. " Direct Identification of Human Gait
   Control", 2015.

This repository contains or links to all of the information needed to reproduce
the results in the paper.

The latest rendered version of the PDF can be viewed via the ShareLaTeX_ CI
system:

.. image:: https://www.sharelatex.com/github/repos/csu-hmc/gait-control-direct-id-paper/builds/latest/badge.svg
   :target: https://www.sharelatex.com/github/repos/csu-hmc/gait-control-direct-id-paper/builds/latest/output.pdf

.. _ShareLaTeX: http://sharelatex.com

License
=======

The content of this repository is licensed under the `Creative Commons
Attribution 4.0 International License`_.

.. image:: https://i.creativecommons.org/l/by/4.0/80x15.png
   :target: http://creativecommons.org/licenses/by/4.0

.. _Creative Commons Attribution 4.0 International License: http://creativecommons.org/licenses/by/4.0

Supplementary Material
======================

The repository contains various modules, scripts, and IPython notebooks that
are included as supplementary material so that readers can explore the analysis
themselves.

Software
--------

The scripts used for the analysis is available in the ``src`` directory of this
repository and depend primarily on two open source Python packages developed
for this paper. The snapshots of the DynamicistToolKit_ 0.3.5 and the
GaitAnalysisToolKit_ 0.1.2 are available via both Zenodo and PyPi:

.. _DynamicistToolKit: http://github.com/moorepants/DynamicistToolKit
.. _GaitAnalysisToolKit: http://github.com/csu-hmc/GaitAnalysisToolKit

Be sure to read the installation instructions for the two packages.

DynamicistToolKit
   .. image:: https://zenodo.org/badge/doi/10.5281/zenodo.13253.svg
      :target: http://dx.doi.org/10.5281/zenodo.13253

   .. image:: https://pypip.in/version/DynamicistToolKit/badge.svg
      :target: https://pypi.python.org/pypi/DynamicistToolKit/
      :alt: Latest Version
GaitAnalysisToolKit
   .. image:: https://zenodo.org/badge/doi/10.5281/zenodo.13159.svg
      :target: http://dx.doi.org/10.5281/zenodo.13159

   .. image:: https://pypip.in/version/GaitAnalysisToolKit/badge.svg
      :target: https://pypi.python.org/pypi/GaitAnalysisToolKit/
      :alt: Latest Version

Furthermore, there are a variety of dependencies that must be installed on your
system to run the scripts. It is best to follow the installation instructions
provided by each of the following software packages for your operating system.

- Various unix tools [#]_: cd, bash, gzip, make, mkdir, rm, tar, unzip, wget
- The `Anaconda Python distribution`_ with Python 2.7 for ease of download and
  management of Python packages.
- Various Python packages: pip, numpy 1.9.1, scipy 0.14.0, matplotlib 1.4.2,
  pytables 3.1.1, pandas 0.15.1, pyyaml 3.11, seaborn 0.5.0, pygments 2.0.1,
  oct2py 2.4.2, DynamicistToolKit 0.3.5, GaitAnalysisToolKit 0.1.2
- Octave_ 3.8.1
- A LaTeX distribution which includes pdflatex. For example: MikTeX_ [Win],
  `TeX Live`_ [Linux], MacTeX_ [Mac].
- Various LaTeX Packages [#]_: minted_, lineno, graphicx, booktabs, cprotect,
  siunitx, inputenc, babel, ifthen, calc, microtype, times, mathptmx, ifpdf,
  amsmath, amsfonts, amssymb, xcolor, authblk, geometry, caption, natbib,
  fancyhdr, lastpage, titlesec, enumitem, bibtex
- Git_ (optional)

.. [#] These are available by default in Linux distributions, provided by Xcode
   on the Mac, and can be obtained via Cygwin, MinGW, or individual install on
   Windows.
.. [#] Most packages will likely be installed with your LaTeX distribution,
   otherwise follow the installation instructions provided by the package. Note
   that minted has abnormal dependencies: Python and Pygments. On Debian based
   systems you will need to install ``texlive-humanities`` and
   ``texlive-science`` to get all of the necessary packages.

.. _Anaconda Python Distribution: http://continuum.io/downloads
.. _Octave: http://octave.org
.. _MikTeX: http://miktex.org
.. _TeX Live: https://www.tug.org/texlive
.. _MacTeX: https://tug.org/mactex
.. _minted: https://github.com/gpoore/minted
.. _Git: http://git-scm.com

Interactive Notebooks
---------------------

The notebooks can be viewed here:

http://nbviewer.ipython.org/github/csu-hmc/gait-control-direct-id-paper/tree/master/notebooks/

Note that the following notebooks are currently not working with the latest
GATK:

- data_munging.ipynb
- perturbations.ipynb

Data
----

The data presented in the paper are available for download from Zenodo under
the Creative Commons CC0 license.

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.13030.svg
   :target: http://dx.doi.org/10.5281/zenodo.13030

Downloading the Repository
==========================

First, navigate to a desired location on your file system and either clone the
repository with Git [#]_ and change into the new directory::

   $ git clone https://github.com/csu-hmc/gait-control-direct-id-paper.git
   $ cd gait-control-direct-id-paper

or download with wget, unpack the zip file, and change into the new directory::

   $ wget https://github.com/csu-hmc/gait-control-direct-id-paper/archive/master.zip
   $ unzip gait-control-direct-id-paper-master.zip
   $ cd gait-control-direct-id-paper-master

.. [#] Please use Git if you wish to contribute back to the repository. See
   CONTRIBUTING.rst for information on how to contribute.

Basic LaTeX Build Instructions
==============================

To build the pdf from the LaTeX source using the pre-generated figures and
tables in the repository, make sure you have an up-to-date LaTeX distribution
installed and run ``make`` from within the repository. The default ``make``
target will build the document, i.e.::

   $ make

You can then view the document with your preferred PDF viewer. For example,
Evince can be used::

   $ evince main.pdf
