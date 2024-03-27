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

`conftest.py <https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files>`_
============================================================================================================================

There should be a ``conftest.py`` file in the ``root/unit`` and ``root/integration``
folders. Additional lower level conftests can be added if it is agreed there
is a need.

`Fixtures <https://docs.pytest.org/en/stable/how-to/fixtures.html#how-to-fixtures>`_
====================================================================================

As far as is possible, the actual test function should do little else but the
actual assertion. Separating off preparation into fixtures may make the code
harder to follow, so compromises are acceptable. For example, setting up a test
``Cube`` should be a fixture, whereas creating a simple string
(``expected = "foo"``), or a single use setup, should *not* be a fixture.


New fixtures should always be considered for conftest when added. If it is
decided that they are not suitably reusable, they can be placed within the
local test file.

`Parameterisation <https://docs.pytest.org/en/stable/example/parametrize.html>`_
================================================================================

Though it is a useful tool, we should not be complicating tests to work around
parameters; they should only be used when it is simple and apparent to implement.

Where you are parameterising multiple tests with the same parameters, it is
usually prudent to use the `parameterisation within fixtures
<https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures>`_.
When doing this, ensure within the tests that it is apparent that they are being
parameterised, either within the fixture name or with comments.

All parameterisation benefits from
`ids <https://docs.pytest.org/en/stable/example/parametrize.html#different-options-for-test-ids>`_,
and so should be used where possible.

`Classes <https://docs.pytest.org/en/stable/getting-started.html#group-multiple-tests-in-a-class>`_
===================================================================================================

How and when to group tests within classes can be based on personal opinion,
we do not deem consistency on this a vital concern.

`Mocks <https://docs.pytest.org/en/stable/how-to/monkeypatch.html>`_
====================================================================

Any mocking should be done with ``pytest.mock``, and monkeypatching where suitable.

.. note::
    If you think we're missing anything important here, please consider creating an
    issue or discussion and share your ideas with the team!

