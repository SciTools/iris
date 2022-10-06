.. include:: ../common_links.inc

.. _iris_development_releases:

Releases
========

A release of Iris is a `tag on the SciTools/Iris`_ Github repository.

The summary below is of the main areas that constitute the release.  The final
section details the :ref:`iris_development_releases_steps` to take.


.. _release_manager:

Release Manager
---------------
A Release Manager will be nominated for each release of Iris. This role involves:

* deciding which features and bug fixes should be included in the release
* managing the project board for the release
* using :discussion:`GitHub Discussion releases category <categories/releases>`
  for documenting intent and capturing any
  discussion about the release

The Release Manager will make the release, ensuring that all the steps outlined
on this page are completed.


Before Release
--------------

Deprecations
~~~~~~~~~~~~

Ensure that any behaviour which has been deprecated for the correct number of
previous releases is now finally changed. More detail, including the correct
number of releases, is in :ref:`iris_development_deprecations`.

Standard Names
~~~~~~~~~~~~~~

Update the file ``etc/cf-standard-name-table.xml`` to the latest CF standard names,
from the `latest CF standard names`_.
( This is used during build to automatically generate the sourcefile
``lib/iris/std_names.py``. )


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

The final steps of the release are to ensure that the release date and details
are correct in the relevant ``whatsnew`` page within the documentation.

There is no need to update the ``iris.__version__``, as this is managed
automatically by `setuptools-scm`_.

Once all checks are complete, the release is published on GitHub by
creating a new tag in the ``SciTools/iris`` repository.


Update conda-forge
------------------

Once a release is cut on GitHub, update the Iris conda recipe on the
`conda-forge iris-feedstock`_ for the release. This will build and publish the
conda package on the `conda-forge Anaconda channel`_.


.. _update_pypi:

Update PyPI
-----------

.. note::

  As part of our Continuous-Integration (CI), the building and publishing of
  PyPI artifacts is now automated by a dedicated GitHub Action.
  
  The following instructions **no longer** require to be performed manually,
  but remain part of the documentation for reference purposes only.

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

After the release is published, the changes from the release branch should be merged
back onto the ``SciTools/iris`` ``main`` branch.

To achieve this, first cut a local branch from the latest ``main`` branch,
and `git merge` the :literal:`.x` release branch into it. Ensure that the
``docs/src/whatsnew/index.rst`` and ``docs/src/whatsnew/latest.rst`` are
correct, before committing these changes and then proposing a pull-request
on the ``main`` branch of ``SciTools/iris``.


Point Releases
--------------

Bug fixes may be implemented and targeted on the :literal:`.x` release branch.
These should lead to a new point release, and another tag.  For example, a fix
for a problem with the ``v1.9.0`` release will be merged into ``v1.9.x`` release
branch, and then released by tagging ``v1.9.1``.

New features shall not be included in a point release, these are for bug fixes.

``whatsnew`` entries should be added to the existing 
``docs/src/whatsnew/v1.9.rst`` file in a new ``v1.9.1`` section. A template for 
this bugfix patches section can be found in the 
``docs/src/whatsnew/latest.rst.template`` file.

A point release does not require a release candidate, but the rest of the
release process is to be followed, including the merge back of changes into
``main``.


.. _iris_development_releases_steps:

Maintainer Steps
----------------

These steps assume a release for ``1.9.0`` is to be created.

Release Steps
~~~~~~~~~~~~~

#. Update the ``whatsnew`` for the release:

   * Use ``git`` to rename ``docs/src/whatsnew/latest.rst`` to the release
     version file ``v1.9.rst``
   * Use ``git`` to delete the ``docs/src/whatsnew/latest.rst.template`` file
   * In ``v1.9.rst`` remove the ``[unreleased]`` caption from the page title.
     Replace this with ``[release candidate]`` for the release candidate and
     remove this for the actual release.
     Note that, the Iris version and release date are updated automatically
     when the documentation is built
   * Review the file for correctness
   * Work with the development team to populate the ``Release Highlights``
     dropdown at the top of the file, which provides extra detail on notable
     changes
   * Use ``git`` to add and commit all changes, including removal of
     ``latest.rst.template``.

#. Update the ``whatsnew`` index ``docs/src/whatsnew/index.rst``

   * Remove the reference to ``latest.rst``
   * Add a reference to ``v1.9.rst`` to the top of the list

#. Check your changes by building the documentation and reviewing
#. Once all the above steps are complete, the release is cut, using
   the :guilabel:`Draft a new release` button on the
   `Iris release page <https://github.com/SciTools/iris/releases>`_
   and targeting the release branch if it exists
#. Create the release feature branch ``v1.9.x`` on `SciTools/iris`_ if it doesn't
   already exist. For point/bugfix releases use the branch which already exists


Post Release Steps
~~~~~~~~~~~~~~~~~~

#. Check the documentation has built on `Read The Docs`_.  The build is
   triggered by any commit to ``main``.  Additionally check that the versions
   available in the pop out menu in the bottom right corner include the new
   release version.  If it is not present you will need to configure the
   versions available in the **admin** dashboard in `Read The Docs`_.
#. Review the `Active Versions`_ for the ``scitools-iris`` project on
   `Read The Docs`_ to ensure that the appropriate versions are ``Active``
   and/or ``Hidden``. To do this ``Edit`` the appropriate version e.g.,
   see `Editing v3.0.0rc0`_ (must be logged into Read the Docs).
#. Merge back to ``main``. This should be done after all releases, including
   the release candidate, and also after major changes to the release branch.
#. On main, make a new ``latest.rst`` from ``latest.rst.template`` and update
   the include statement and the toctree in ``index.rst`` to point at the new
   ``latest.rst``.


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
.. _latest CF standard names: http://cfconventions.org/standard-names.html
.. _setuptools-scm: https://github.com/pypa/setuptools_scm
