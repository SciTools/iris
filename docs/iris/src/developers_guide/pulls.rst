Pull Request Check List
***********************

A pull request to a SciTools project master should be ready to merge into the
master branch.

All pull request will be reviewed by a core developer who will manage the
process of merging. It is the responsibility of a developer submitting a
pull request to do their best to deliver a pull request which meets the
requirements of the project it is submitted to. 

The check list summarises criteria which will be checked before a pull request
is merged.  Before submitting a pull request please consider this list.


The Iris Check List
====================

* Have you provided a helpful description of the Pull Request? What has 
  changed and why.  This should include:

 * the aim of the change - the problem addressed, a link to the issue;
 * how the change has been delivered.

* Do new files pass PEP8?

 * PEP8_ is the Python source code style guide.
 * There is a python module for checking pep8 compliance: python-pep8_

* Do all the tests pass locally?

 * The Iris tests may be run with ``python setup.py test`` which has a command 
   line utility included.
 * Coding standards, including PEP8_ compliance and copyright message (including 
   the correct year of the latest change), are tested. 

* Has a new test been provided?

* Has iris-test-data been updated?

 * iris-test-data_ is a github project containing all the data to support the
   tests.
 * If this has been updated a reference to the relevant pull request should be
   provided.

* Has the the documentation been updated to explain the new feature or bug fix?

 * with reference to the developer guide on docstrings_

* Have code examples been provided inside the relevant docstrings?

* Has iris-sample-data been updated?

 * iris-sample-data_ is a github project containing all the data to support
   the gallery and examples.

* Does the documentation build without errors?

 * The documentation is built using ``make html`` in ``./docs/iris``.

* Do the documentation tests pass?

 * ``make doctest``, ``make extest``  in ``./docs/iris``.

* Is there an associated iris-code-generators pull request?

 * iris-code-generators_ is a github project which provides processes for
   generating a small subset of the Iris source code files from other 
   information sources.

* Has the travis file been updated to reflect any dependency updates?

 * ``./.travis.yml`` is used to manage the continuous integration testing.


.. _PEP8: http://www.python.org/dev/peps/pep-0008/
.. _python-pep8: https://pypi.python.org/pypi/pep8
.. _iris-test-data: https://github.com/SciTools/iris-test-data
.. _iris-sample-data: https://github.com/SciTools/iris-sample-data
.. _iris-code-generators: https://github.com/SciTools/iris-code-generators
.. _docstrings: http://scitools.org.uk/iris/docs/latest/developers_guide/documenting/docstrings.html
