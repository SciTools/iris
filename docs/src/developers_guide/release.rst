.. include:: ../common_links.inc

.. _iris_development_releases:

Releases
========

A release of Iris is a `tag on the SciTools/Iris`_ Github repository.

Below is :ref:`iris_development_releases_steps`, followed by some prose on the
main areas that constitute the release.


.. _iris_development_releases_steps:

How to Create an Iris Release
-----------------------------

The step-by-step process is walked-through by a script at:
``<Iris repo root>/tools/release_do_nothing.py``, and also available here:
:doc:`release_do_nothing`.


.. _release_manager:

Release Manager
---------------

A Release Manager will be nominated for each release of Iris. This role involves:

* deciding which features and bug fixes should be included in the release
* managing the `GitHub Projects`_ board for the release
* using :discussion:`GitHub Discussion releases category <categories/releases>`
  for documenting intent and capturing any
  discussion about the release
* holding a developer retrospective post release, to look for potential
  future improvements

The Release Manager will make the release, ensuring that all the steps outlined
on this page are completed.


Versioning
----------

Iris' version numbers conform to `Semantic Versioning`_ (``MAJOR.MINOR.PATCH``)
and `PEP 440`_.

Iris uses `setuptools-scm`_ to automatically manage versioning based on Git
tags. No manual versioning work is required within the files themselves.


Release Candidate
-----------------

Prior to a release, a release candidate tag may be created, marked as a
pre-release in GitHub, with a tag ending with :literal:`rc` followed by a
number (0-based), e.g.,:

    :literal:`v1.9.0rc0`

If created, the pre-release shall be available for a minimum of 2 weeks
prior to the release being cut.  However a 4 week period should be the goal
to allow user groups to be notified of the existence of the pre-release and
encouraged to test the functionality.

A pre-release is expected for a major or minor release, but not for a
patch release.

If new features are required for a release after a release candidate has been
cut, a new pre-release shall be issued first.

Release candidates are made available as a conda package on the
`conda-forge Anaconda channel`_ using the `rc_iris`_ label. This is achieved via
the `conda-forge iris-feedstock`_ following `CFEP-05`_. For further information
see the `conda-forge User Documentation`_.


Patch Releases
--------------

Patch releases may be implemented to fix problems with previous major or minor
releases. E.g. ``v1.9.1`` to fix a problem in ``v1.9.0``, both being part of
the ``v1.9`` series.

New features shall not be included in a patch release, these are for bug fixes.

A patch release does not require a release candidate, but the rest of the
release process is to be followed.


Before Release
--------------

Deprecations
~~~~~~~~~~~~

Any behaviour which has been deprecated for the correct number of
previous releases is now finally changed. More detail, including the correct
number of releases, is in :ref:`iris_development_deprecations`.

Standard Names
~~~~~~~~~~~~~~

The file ``etc/cf-standard-name-table.xml`` is updated to the latest CF standard names,
from the `latest CF standard names`_.
( This is used during build to automatically generate the sourcefile
``lib/iris/std_names.py``. )


The Release
-----------

Release Branch
~~~~~~~~~~~~~~

Once the features intended for the release are on ``main``, a release branch
should be created, in the ``SciTools/iris`` repository.  This will have the name:

    :literal:`v{major release number}.{minor release number}.x`

for example:

    :literal:`v1.9.x`

This branch shall be used to finalise the release details in preparation for
the release candidate.

Changes for a **patch release** should target to the same release branch as the
rest of the series. For example, a fix
for a problem with the ``v1.9.0`` release will be merged into ``v1.9.x`` release
branch, and then released with the tag ``v1.9.1``.

Documentation
~~~~~~~~~~~~~

The documentation should include a dedicated What's New file for this release
series (e.g. ``v1.9.rst``), incorporating all of the What's New entries for the release.
This content should be reviewed and adapted as required, including highlights
at the top of the What's New document.

What's New entries for **patch releases** should be added to the existing file
for that release series (e.g. ``v1.9.1`` section in the ``v1.9.rst`` file).

A template for What's New formatting can be found in the
``docs/src/whatsnew/latest.rst.template`` file.


Tagging
~~~~~~~

Once all checks are complete, the release is published from the release
branch - via the GitHub release functionality in the ``SciTools/iris``
repository - which simultaneously creates a Git tag for the release.


Post Release
------------

PyPI
~~~~
Iris is available on PyPI as ``scitools-iris``.

Iris' Continuous-Integration (CI) includes the automatic building and publishing of
PyPI artifacts in a dedicated GitHub Action.

Legacy manual instructions are appended to this page for reference purposes
(:ref:`update_pypi`)

conda-forge
~~~~~~~~~~~

Iris is available on conda-forge as ``iris``.

This is managed via the the Iris conda recipe on the
`conda-forge iris-feedstock`_, which is updated after the release is cut on
GitHub, followed by automatic build and publish of the
conda package on the `conda-forge Anaconda channel`_.

Announcement
~~~~~~~~~~~~

Iris uses Bluesky (`@scitools.bsky.social`_) to announce new releases, as well as any
internal message boards that are accessible (e.g. at the UK Met Office).
Announcements usually include a highlighted feature to hook readers' attention.

Citation
~~~~~~~~

``docs/src/userguide/citation.rst`` is updated to include
the latest [non-release-candidate] version, date and `Zenodo DOI`_
of the new release. Ideally this would be updated before the release, but
the DOI for the new version is only available once the release has been
created in GitHub.

Merge Back
~~~~~~~~~~

After any release is published, **including patch releases**, the changes from the
release branch should be merged back onto the ``SciTools/iris`` ``main`` branch.


Appendices
----------

.. _update_pypi:

Updating PyPI Manually
~~~~~~~~~~~~~~~~~~~~~~

.. note::

  As part of our Continuous-Integration (CI), the building and publishing of
  PyPI artifacts is now automated by a dedicated GitHub Action.

  The following instructions **no longer** require to be performed manually,
  but remain part of the documentation for reference purposes only.

Update the `scitools-iris`_ project on PyPI with the latest Iris release.

To do this perform the following steps.

Create a conda environment with the appropriate conda packages to build the
source distribution (``sdist``) and pure Python wheel (``bdist_wheel``)::

    > conda create -n iris-pypi -c conda-forge --yes build twine
    > . activate iris-pypi

Checkout the appropriate Iris ``<release>`` tag from the appropriate ``<repo>``.
For example, to checkout tag ``v1.0`` from ``upstream``::

    > git fetch upstream --tags
    > git checkout v1.0

Build the source distribution and wheel from the Iris root directory::

    > python -m build

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
    > conda env create --file ./requirements/iris.yml
    > . activate iris-dev
    > python -m pip install --no-deps scitools-iris

For further details on how to test Iris, see :ref:`developer_running_tests`.

.. seealso::

    For further information on packaging and uploading a project to PyPI, please
    refer to `Generating Distribution Archives`_ and `Packaging Your Project`_.

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
.. _latest CF standard names: https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml
.. _setuptools-scm: https://github.com/pypa/setuptools_scm
.. _Semantic Versioning: https://semver.org/
.. _PEP 440: https://peps.python.org/pep-0440/
.. _@scitools.bsky.social: https://bsky.app/profile/scitools.bsky.social
.. _GitHub Projects: https://github.com/SciTools/iris/projects
.. _Zenodo DOI: https://doi.org/10.5281/zenodo.595182
