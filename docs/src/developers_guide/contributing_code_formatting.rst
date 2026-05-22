.. include:: ../common_links.inc

.. _code_formatting:

Code Formatting
===============

Also known as 'linting'. This is used to ensure a consistent code format
throughout Iris, maximising the maintainability and quality. Code formatting
is checked using the `pre-commit`_ tool, and the full list current formatting
tools is defined in Iris' `pre-commit-config.yaml`_ file. Read more about
linting on the `SciTools wiki page`_.

``pre-commit`` compliance is automatically checked on all Iris pull requests
(more info: :ref:`pre_commit_ci`), but you can also run pre-commit locally as
Git hooks - every time you make a commit:

You can install ``pre-commit`` in your development environment using ``pip``::

    $ pip install pre-commit

or alternatively using ``conda``::

    $ conda install -c conda-forge pre-commit

.. note:: If you have setup your Python environment using the guide
         :ref:`installing_from_source` then ``pre-commit`` should already
         be present.

In order to install the ``pre-commit`` git hooks defined in our
``.pre-commit-config.yaml`` file, you must now run the following command from
the root directory of Iris::

    $ pre-commit install

Upon performing a ``git commit``, your code changes will be automatically
checked against all Iris' ``pre-commit`` hooks. For some hooks this includes
automated edits of your code e.g. formatting or sorting of imports; these new
edits are not staged for you - i.e. you need to run ``git add`` again on that
file. Note that,
``pre-commit`` will automatically download and install the necessary packages
for each ``.pre-commit-config.yaml`` git hook.

With the ``pre-commit`` configured, the output of performing a ``git commit``
will look similar to::

    Check for added large files..............................................Passed
    Check for merge conflicts................................................Passed
    Debug Statements (Python)............................(no files to check)Skipped
    Don't commit to branch...................................................Passed
    black................................................(no files to check)Skipped
    flake8...............................................(no files to check)Skipped
    [contribution_overhaul c8513187] this is my commit message
    2 files changed, 10 insertions(+), 9 deletions(-)


.. _type_hinting:

Type Hinting
------------
Iris is gradually adding
`type hints <https://docs.python.org/3/library/typing.html>`_ into the
codebase. The reviewer will look for type hints in a pull request; if you're
not confident with these, feel free to work together with the reviewer to
add/improve them.


.. _pre-commit: https://pre-commit.com/
.. _pre-commit-config.yaml: https://github.com/SciTools/iris/blob/main/.pre-commit-config.yaml
.. _SciTools wiki page: https://github.com/SciTools/.github/wiki/Linting
