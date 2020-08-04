.. _iris_development_releases:

Releases
========

A release of Iris is a `tag on the SciTools/Iris`_ 
Github repository.

The summary below is of the main areas that constitute the release.  The final
section details the :ref:`iris_development_releases_steps` to take.


Release branch
--------------

Once the features intended for the release are on master, a release branch 
should be created, in the SciTools/Iris repository.  This will have the name:

    :literal:`v{major release number}.{minor release number}.x`

for example:

    :literal:`v1.9.x`

This branch shall be used to finalise the release details in preparation for
the release candidate.


Release candidate
-----------------

Prior to a release, a release candidate tag may be created, marked as a
pre-release in github, with a tag ending with :literal:`rc` followed by a
number, e.g.:

    :literal:`v1.9.0rc1`

If created, the pre-release shall be available for a minimum of two weeks 
prior to the release being cut.  However a 4 week period should be the goal
to allow user groups to be notified of the existence of the pre-release and
encouraged to test the functionality.

A pre-release is expected for a minor release, but will not for a
point release.

If new features are required for a release after a release candidate has been
cut, a new pre-release shall be issued first.


Documentation
-------------

The documentation should include all of the what's new entries for the release.
This content should be reviewed and adapted as required.

Steps to achieve this can be found in the :ref:`iris_development_releases_steps`.


The release
-----------

The final steps are to change the version string in the source of 
:literal:`Iris.__init__.py` and include the release date in the relevant what's
new page within the documentation.

Once all checks are complete, the release is cut by the creation of a new tag
in the SciTools Iris repository.


Conda recipe
------------

Once a release is cut, the `Iris feedstock`_ for the conda recipe must be
updated to build the latest release of Iris and push this artefact to
`conda forge`_.  

.. _Iris feedstock: https://github.com/conda-forge/iris-feedstock/tree/master/recipe
.. _conda forge: https://anaconda.org/conda-forge/iris

Merge back
----------

After the release is cut, the changes shall be merged back onto the
Scitools/iris master branch.

To achieve this, first cut a local branch from the release branch,
:literal:`{release}.x`.  Next add a commit changing the release string to match
the release string on scitools/master.  This branch can now be proposed as a
pull request to master.  This work flow ensures that the commit identifiers are
consistent between the :literal:`.x` branch and :literal:`master`.


Point releases
--------------

Bug fixes may be implemented and targeted as the :literal:`.x` branch.  These
should lead to a new point release, another tag.  For example, a fix for a
problem with 1.9.0 will be merged into 1.9.x, and then released by tagging
1.9.1.

New features shall not be included in a point release, these are for bug fixes.

A point release does not require a release candidate, but the rest of the
release process is to be followed, including the merge back of changes into
:literal:`master`.  


.. _iris_development_releases_steps:

Maintainer steps
----------------

These steps assume a release for ``v1.9`` is to be created

Release steps
~~~~~~~~~~~~~

#. Create the branch ``1.9.x`` on the main repo, not in a forked repo, for the
   release candidate or release.  The only exception is for a point/bugfix
   release as it should already exist
#. Update the what's new for the release:  

    * Copy ``docs/iris/src/whatsnew/latest.rst`` to a file named
      ``v1.9.rst``
    * Delete the ``docs/iris/src/whatsnew/latest.rst`` file so it will not
      cause an issue in the build
    * In ``v1.9.rst`` update the page title (first line of the file) to show
      the date and version in the format of ``v1.9 (DD MMM YYYY)``.  For
      example ``v1.9 (03 Aug 2020)``
    * Review the file for correctness
    * Add ``v1.9.rst`` to git and commit all changes, including removal of
      ``latest.rst``

#. Update the what's new index ``docs/iris/src/whatsnew/index.rst``

   * Temporarily remove reference to ``latest.rst``
   * Add a reference to ``v1.9.rst`` to the top of the list

#. Update the ``Iris.__init__.py`` version string, to ``1.9.0``
#. Check your changes by building the documentation and viewing the changes
#. Once all the above steps are complete, the release is cut, using 
   the :guilabel:`Draft a new release` button on the
   `Iris release page <https://github.com/SciTools/iris/releases>`_


Post release steps
~~~~~~~~~~~~~~~~~~

#. Check the documentation has built on `Read The Docs`_.  The build is 
   triggered by any commit to master.  Additionally check that the versions
   available in the pop out menu in the bottom left corner include the new
   release version.  If it is not present you will need to configure the
   versions avaiable in the **admin** dashboard in Read The Docs
#. Copy ``docs/iris/src/whatsnew/latest.rst.template`` to 
   ``docs/iris/src/whatsnew/latest.rst``.  This will reset
   the file with the ``unreleased`` heading and placeholders for the what's
   new headings
#. Add back in the reference to ``latest.rst`` to the what's new index 
   ``docs/iris/src/whatsnew/index.rst``
#. Update ``Iris.__init__.py`` version string to show as ``1.10.dev0``
#. Merge back to master


.. _Read The Docs: https://readthedocs.org/projects/scitools-iris/builds/
.. _tag on the SciTools/Iris: https://github.com/SciTools/iris/releases
