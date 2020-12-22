.. include:: ../../common_links.inc

.. _reST_quick_start:

================
reST Quick Start
================

`reST`_ is used to create the documentation for Iris_.  It is used to author
all of the documentation content including use in docstrings where appropriate.
For more information see :ref:`docstrings`.

reST is a lightweight markup language intended to be highly readable in
source format. This guide will cover some of the more frequently used advanced
reST markup syntaxes, for the basics of reST the following links may be useful:

 * https://www.sphinx-doc.org/en/master/usage/restructuredtext/
 * http://packages.python.org/an_example_pypi_project/sphinx.html

Reference documentation for reST can be found at http://docutils.sourceforge.net/rst.html.

Creating Links
--------------
Basic links can be created with ```Text of the link <http://example.com>`_``
which will look like `Text of the link <http://example.com>`_


Documents in the same project can be cross referenced with the syntax
``:doc:`document_name``` for example, to reference the "docstrings" page
``:doc:`docstrings``` creates the following link :doc:`docstrings`


References can be created between sections by first making a "label" where
you would like the link to point to ``.. _name_of_reference::`` the
appropriate link can now be created with ``:ref:`name_of_reference```
(note the trailing underscore on the label)


Cross referencing other reference documentation can be achieved with the
syntax ``:py:class:`zipfile.ZipFile``` which will result in links such as
:py:class:`zipfile.ZipFile` and :py:class:`numpy.ndarray`.


.. _reST: http://en.wikipedia.org/wiki/ReStructuredText
