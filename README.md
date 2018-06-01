<h1 align="center">
  <a href="https://scitools.org.uk/iris/docs/latest/" style="display: block; margin: 0 auto;">
   <img src="https://raw.githubusercontent.com/pelson/iris/markdown_readme/docs/iris/src/_static/logo_banner.png"
        style="max-width: 40%;" alt="Iris"></a><br>
</h1>

<h4 align="center">
    Iris is a powerful, easy to use, and community-driven Python library for
    analysing and visualising Earth science data
</h4>

<p align="center">
<!-- https://shields.io/ is a good source of these -->
<a href="https://anaconda.org/conda-forge/iris">
<img src="https://img.shields.io/conda/dn/conda-forge/iris.svg"
     alt="conda-forge downloads" /></a>
<a href="https://github.com/SciTools/iris/releases">
<img src="https://img.shields.io/github/tag/SciTools/iris.svg"
     alt="Latest version" /></a>
<a href="https://github.com/SciTools/iris/commits/master">
<img src="https://img.shields.io/github/commits-since/SciTools/iris/latest.svg"
     alt="Commits since last release" /></a>
<a href="https://github.com/SciTools/iris/graphs/contributors">
<img src="https://img.shields.io/github/contributors/SciTools/iris.svg"
     alt="# contributors" /></a>
<a href="https://travis-ci.org/SciTools/iris/branches">
<img src="https://api.travis-ci.org/repositories/SciTools/iris.svg?branch=master"
     alt="Travis-CI" /></a>
<a href="https://zenodo.org/badge/latestdoi/5312648">
<img src="https://zenodo.org/badge/5312648.svg"
     alt="zenodo" /></a>
</p>
<br>

<!-- NOTE: toc auto-generated with https://github.com/frnmst/md-toc:
   $ md_toc github README.md -i
-->

<h1>Table of contents</h1>

[](TOC)

+ [Overview](#overview)
+ [Documentation](#documentation)
+ [Installation](#installation)
+ [Copyright and licence](#copyright-and-licence)

[](TOC)

# Overview

Iris implements a data model based on the [CF conventions](http://cfconventions.org/)
giving you a powerful, format-agnostic, interface for working with your data.
It excels when working with multi-dimensional Earth Science data, where tabular
representations become unwieldy and inefficient.

[CF Standard names](http://cfconventions.org/standard-names.html),
[units](https://github.com/SciTools/cf_units), and coordinate metadata
are built-in to Iris, giving you a rich and expressive interface for maintaining
an accurate representation of your data. Its first-class treatment of data and
associated metadata, includes:

  * aggregations and reductions (min, max, (area-)weighted mean, etc.)
  * interpolation and regridding (nearest-neighbor, linear, area-weighted, etc.)
  * operator overloads (``+``, ``-``, ``*``, ``/``, etc.)
  * merge and concatenate
  * subsetting and extraction
  * unit conversion
  * a visualisation interface based on [matplotlib](https://matplotlib.org/) and
    [cartopy](https://scitools.org.uk/cartopy/docs/latest/)

A number of file formats are recognised by Iris, including CF-compliant NetCDF, GRIB,
and PP, and it has a plugin architecture to allow other formats to be added seamlessly.

Building upon [numpy](http://www.numpy.org/) and [dask](https://dask.pydata.org/en/latest/),
Iris scales from efficient single-machine workflows right through to multi-core clusters and HPC.
Interoperability with packages from the wider scientific python ecosystem comes from Iris'
use of standard numpy/dask arrays as its underlying data storage.


# Documentation

The documentation for Iris is available at <https://scitools.org.uk/iris/docs/latest>,
including a user guide, example code, and gallery.

# Installation

The easiest way to install Iris is with [conda](https://conda.io/miniconda.html):

    conda install -c conda-forge iris

Detailed instructions, including information on installing from source,
are available in [INSTALL](INSTALL).


# Copyright and licence

Iris may be freely distributed, modified and used commercially under the terms
of its [GNU LGPLv3 license](COPYING.LESSER).


(C) British Crown Copyright 2010 - 2018, Met Office
