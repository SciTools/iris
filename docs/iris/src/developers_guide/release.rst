.. _iris_development_releases:

Releases
********

A release of Iris is a tag on the SciTools/Iris Github repository.

Release Branch
==============

Once the features intended for the release are on master, a release branch should be created, in the SciTools/Iris repository.  This will have the name:

    :literal:`{major release number}.{minor release number}.x`

for example:

    :literal:`v1.9.x`

This branch shall be used to finalise the release details in preparation for the release candidate.

Release Candidate
=================

Prior to a release, a release candidate tag may be created, marked as a pre-release in github, with a tag ending with :literal:`rc` followed by a number, e.g.:

    :literal:`v1.9.0rc1`

If created, the pre-release shall be available for at least one week prior to the release being cut.  User groups should be notified of the existence of the pre-release and encouraged to test the functionality.

A pre-release is expected for a minor release, but not normally provided for a point release.

If new features are required for a release after a release candidate has been cut, a new pre-release shall be issued first.

Documentation
=============

The documentation should include all of the what's new snippets, which must be compiled into a what's new.  This content should be reviewed and adapted as required and the snippets removed from the branch to produce a coherent what's new page.

Upon release, the documentation shall be added to the SciTools scitools.org.uk github project's gh-pages branch as the latest documentation.

Testing the Conda Recipe
========================

Before a release is cut, the SciTools conda-recipes-scitools recipe for Iris shall be tested to build the release branch of Iris; this test recipe shall not be merged onto conda-recipes-scitools.

The Release
===========

The final steps are to change the version string in the source of :literal:`Iris.__init__.py` and include the release date in the relevant what's new page within the documentation.

Once all checks are complete, the release is cut by the creation of a new tag in the SciTools Iris repository.

Conda Recipe
============

Once a release is cut, the SciTools conda-recipes-scitools recipe for Iris shall be updated to build the latest release of Iris and push this artefact to anaconda.org.  The build and push is all automated as part of the merge process.

Merge Back
==========

After the release is cut, the changes shall be merged back onto the scitools master.

To achieve this, first cut a local branch from the release branch, :literal:`{release}.x`.  Next add a commit changing the release string to match the release string on scitools/master.  
This branch can now be proposed as a pull request to master.  This work flow ensures that the commit identifiers are consistent between the :literal:`.x` branch and :literal:`master`.

Point Releases
==============

Bug fixes may be implemented and targeted as the :literal:`.x` branch.  These should lead to a new point release, another tag.
For example, a fix for a problem with 1.9.0 will be merged into 1.9.x, and then released by tagging 1.9.1.

New features shall not be included in a point release, these are for bug fixes.

A point release does not require a release candidate, but the rest of the release process is to be followed, including the merge back of changes into :literal:`master`.  

