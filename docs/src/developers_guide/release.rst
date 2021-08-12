.. include:: ../common_links.inc

.. _iris_development_releases:

Releases
========

A release of Iris is a `tag on the SciTools/Iris`_ Github repository.

The summary below is of the main areas that constitute the release.  The final
section details the :ref:`iris_development_releases_steps` to take.


Before Release
--------------

Deprecations
~~~~~~~~~~~~

Ensure that any behaviour which has been deprecated for the correct number of
previous releases is now finally changed. More detail, including the correct
number of releases, is in :ref:`iris_development_deprecations`.


Release Branch
--------------

Once the features intended for the release are on ``main``, a release branch
should be created, in the ``SciTools/iris`` repository.  This will have the name:

    :literal:`v{major release number}.{minor release number}.x`

for example:

    :literal:`v1.9.x`

This branch shall be used to finalise the release details in preparation for
the release candidate.


Release Candidate
-----------------

Prior to a release, a release candidate tag may be created, marked as a
pre-release in GitHub, with a tag ending with :literal:`rc` followed by a
number (0-based), e.g.,:

    :literal:`v1.9.0rc0`

If created, the pre-release shall be available for a minimum of two weeks
prior to the release being cut.  However a 4 week period should be the goal
to allow user groups to be notified of the existence of the pre-release and
encouraged to test the functionality.

A pre-release is expected for a major or minor release, but not for a
point release.

If new features are required for a release after a release candidate has been
cut, a new pre-release shall be issued first.

Make the release candidate available as a conda package on the
`conda-forge Anaconda channel`_ using the `rc_iris`_ label. To do this visit
the `conda-forge iris-feedstock`_ and follow `CFEP-05`_. For further information
see the `conda-forge User Documentation`_.


Documentation
-------------

The documentation should include all of the ``whatsnew`` entries for the release.
This content should be reviewed and adapted as required.

Steps to achieve this can be found in the :ref:`iris_development_releases_steps`.


The Release
-----------

The final steps of the release are to change the version string ``__version__``
in the source of :literal:`iris.__init__.py` and ensure the release date and details
are correct in the relevant ``whatsnew`` page within the documentation.

Once all checks are complete, the release is cut by the creation of a new tag
in the ``SciTools/iris`` repository.


Update conda-forge
------------------

Once a release is cut on GitHub, update the Iris conda recipe on the
`conda-forge iris-feedstock`_ for the release. This will build and publish the
conda package on the `conda-forge Anaconda channel`_.


.. _update_pypi:

Update PyPI
-----------

Update the `scitools-iris`_ project on PyPI with the latest Iris release.

To do this perform the following steps.

Create a conda environment with the appropriate conda packages to build the
source distribution (``sdist``) and pure Python wheel (``bdist_wheel``)::

    > conda create -n iris-pypi -c conda-forge --yes pip python setuptools twine wheel
    > . activate iris-pypi

Checkout the appropriate Iris ``<release>`` tag from the appropriate ``<repo>``.
For example, to checkout tag ``v1.0`` from ``upstream``::

    > git fetch upstream --tags
    > git checkout v1.0 

Build the source distribution and wheel from the Iris root directory::

    > python setup.py sdist bdist_wheel

This ``./dist`` directory should now be populated with the source archive
``.tar.gz`` file, and built distribution ``.whl`` file.

Check that the package description will render properly on PyPI for each
of the built artifacts::

    > python -m twine check dist/*

To list and check the contents of the binary wheel::

    > python -m zipfile --list dist/*.whl

If all seems well, sufficient maintainer privileges will be required to
upload these artifacts to `scitools-iris`_ on PyPI::

    > python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

Ensure that the artifacts are successfully uploaded and available on
`scitools-iris`_ before creating a conda test environment to install Iris
from PyPI::

    > conda deactivate
    > conda env create --file ./requrements/ci/iris.yml
    > . activate iris-dev
    > conda install -c conda-forge pip
    > python -m pip install --no-deps scitools-iris

For further details on how to test Iris, see :ref:`developer_running_tests`.

.. seealso::

    For further information on packaging and uploading a project to PyPI, please
    refer to `Generating Distribution Archives`_ and `Packaging Your Project`_.


Merge Back
----------

After the release is cut, the changes from the release branch should be merged
back onto the ``SciTools/iris`` ``main`` branch.

To achieve this, first cut a local branch from the latest ``main`` branch,
and `git merge` the :literal:`.x` release branch into it. Ensure that the
``iris.__version__``, ``docs/src/whatsnew/index.rst`` and ``docs/src/whatsnew/latest.rst``
are correct, before committing these changes and then proposing a pull-request
on the ``main`` branch of ``SciTools/iris``.


Point Releases
--------------

Bug fixes may be implemented and targeted on the :literal:`.x` release branch.
These should lead to a new point release, and another tag.  For example, a fix
for a problem with the ``v1.9.0`` release will be merged into ``v1.9.x`` release
branch, and then released by tagging ``v1.9.1``.

New features shall not be included in a point release, these are for bug fixes.

A point release does not require a release candidate, but the rest of the
release process is to be followed, including the merge back of changes into
``main``.


.. _iris_development_releases_steps:

Maintainer Steps
----------------

These steps assume a release for ``1.9.0`` is to be created.

Release Steps
~~~~~~~~~~~~~

#. Create the release feature branch ``v1.9.x`` on `SciTools/iris`_.
   The only exception is for a point/bugfix release, as it should already exist
#. Update the ``iris.__init__.py`` version string e.g., to ``1.9.0``
#. Update the ``whatsnew`` for the release:

    * Use ``git`` to rename ``docs/src/whatsnew/latest.rst`` to the release
      version file ``v1.9.rst``
    * Use ``git`` to delete the ``docs/src/whatsnew/latest.rst.template`` file
    * In ``v1.9.rst`` remove the ``[unreleased]`` caption from the page title.
      Note that, the Iris version and release date are updated automatically
      when the documentation is built
    * Review the file for correctness
    * Work with the development team to populate the ``Release Highlights``
      dropdown at the top of the file, which provides extra detail on notable
      changes
    * Use ``git`` to add and commit all changes, including removal of
      ``latest.rst.template``

#. Update the ``whatsnew`` index ``docs/src/whatsnew/index.rst``

   * Remove the reference to ``latest.rst``
   * Add a reference to ``v1.9.rst`` to the top of the list

#. Check your changes by building the documentation and reviewing
#. Once all the above steps are complete, the release is cut, using
   the :guilabel:`Draft a new release` button on the
   `Iris release page <https://github.com/SciTools/iris/releases>`_


Post Release Steps
~~~~~~~~~~~~~~~~~~

#. Check the documentation has built on `Read The Docs`_.  The build is
   triggered by any commit to ``main``.  Additionally check that the versions
   available in the pop out menu in the bottom left corner include the new
   release version.  If it is not present you will need to configure the
   versions available in the **admin** dashboard in `Read The Docs`_.
#. Review the `Active Versions`_ for the ``scitools-iris`` project on
   `Read The Docs`_ to ensure that the appropriate versions are ``Active``
   and/or ``Hidden``. To do this ``Edit`` the appropriate version e.g.,
   see `Editing v3.0.0rc0`_ (must be logged into Read the Docs).
#. Copy ``docs/src/whatsnew/latest.rst.template`` to
   ``docs/src/whatsnew/latest.rst``.  This will reset
   the file with the ``unreleased`` heading and placeholders for the
   ``whatsnew`` headings
#. Add back in the reference to ``latest.rst`` to the ``whatsnew`` index
   ``docs/src/whatsnew/index.rst``
#. Update ``iris.__init__.py`` version string to show as ``1.10.dev0``
#. Merge back to ``main``


.. _SciTools/iris: https://github.com/SciTools/iris
.. _tag on the SciTools/Iris: https://github.com/SciTools/iris/releases
.. _conda-forge Anaconda channel: https://anaconda.org/conda-forge/iris
.. _conda-forge iris-feedstock: https://github.com/conda-forge/iris-feedstock
.. _CFEP-05: https://github.com/conda-forge/cfep/blob/master/cfep-05.md
.. _conda-forge User Documentation: https://conda-forge.org/docs/user/00_intro.html
.. _Active Versions: https://readthedocs.org/projects/scitools-iris/versions/
.. _Editing v3.0.0rc0: https://readthedocs.org/dashboard/scitools-iris/version/v3.0.0rc0/edit
.. _rc_iris: https://anaconda.org/conda-forge/iris/labels
.. _Generating Distribution Archives: https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives
.. _Packaging Your Project: https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project
