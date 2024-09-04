.. include:: ../../common_links.inc
.. _docstrings:

==========
Docstrings
==========

Every public object in the Iris package should have an appropriate docstring.
This is important as the docstrings are used by developers to understand
the code and may be read directly in the source or via the
:doc:`Iris API </generated/api/iris/index>`.

.. note::
   As of April 2022 we are looking to adopt `numpydoc`_ strings as standard.
   We aim to complete the adoption over time as we do changes to the codebase.
   For examples of use see `numpydoc`_ and `sphinxcontrib-napoleon`_

For consistency always use:

* ``"""triple double quotes"""`` around docstrings.
* ``r"""raw triple double quotes"""`` if you use any backslashes in your
  docstrings.
* ``u"""Unicode triple-quoted string"""`` for Unicode docstrings

All docstrings can use reST (reStructuredText) markup to augment the
rendered formatting.  See the :ref:`reST_quick_start` for more detail.

For more information including examples pleasee see:

* `numpydoc`_
* `sphinxcontrib-napoleon`_


.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#style-guide
.. _sphinxcontrib-napoleon: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html