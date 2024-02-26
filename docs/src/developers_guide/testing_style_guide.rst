.. include:: ../common_links.inc

.. _testing_style_guide:

PyTest Style Guide
******************

This style guide should be approached pragmatically. Most of the guidelines laid out
below will not be practical in every scenario, and as such should not be considered
firm rules.

At time of writing, some existing tests have already been written in PyTest,
so might not be abiding by these guidelines.

Directory
=========

Where suitable, tests should be located within the relevant directories.
In most circumstance, that means new tests should not be placed in the
root test directory, but in the relevant sub-folders.

Conftest.py
===========

There should be a conftest.py file in the root/unit and root/integration
folders. Additional lower level conftests can be added if it is agreed there
is a need.

Fixtures
========

As far as is possible, the actual test function should do little else but the
actual assertion. However, in some cases this will not be applicable, so this
will have to be decided on a case by case basis.

New fixtures should always be considered for conftest when added. If it is
decided that they are not suitably reusable, they can be placed within the
local test file.

Parameterisation
================

Though it is a useful tool, we should not be complicating tests to work around
parameters; they should only be used when it is simple and apparent to implement.

Where you are parameterising multiple tests with the same parameters, it is
usually prudent to use the parameterisation within fixtures. When doing this,
ensure within the tests that it is apparent it's being parameterised,
either within the fixture name or with comments.

All parameterisation benefits from ids, and so should be used where possible.

Classes
=======

How and when to group tests within classes can be based on personal opinion,
we do not deem consistency on this a vital concern.

Mocks
=====

Any mocking should be done with pytest.mock, and monkeypatching where suitable.

.. note::
    If you think we're missing anything important here, please consider creating an
    issue or discussion and share your ideas with the team!

