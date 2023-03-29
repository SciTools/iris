.. include:: ../common_links.inc

.. _code_formatting:

Code Formatting
===============

To ensure a consistent code format throughout Iris, we recommend using
tools to check the source directly.

* `black`_ for an opinionated coding auto-formatter
* `flake8`_ linting checks

The preferred way to run these tools automatically is to setup and configure
`pre-commit`_.

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

Upon performing a ``git commit``, your code will now be automatically formatted
to the ``black`` configuration defined in our ``pyproject.toml`` file, and
linted according to our ``.flake8`` configuration file. Note that,
``pre-commit`` will automatically download and install the necessary packages
for each ``.pre-commit-config.yaml`` git hook.

Additionally, you may wish to enable ``black`` for your preferred
`editor/IDE <https://black.readthedocs.io/en/stable/integrations/editors.html#editor-integration>`_.

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


.. note:: You can also run `black`_ and `flake8`_ manually.  Please see the
          their officially documentation for more information.

Type Hinting
------------
Iris is gradually adding
`type hints <https://docs.python.org/3/library/typing.html>`_ into the
codebase. The reviewer will look for type hints in a pull request; if you're
not confident with these, feel free to work together with the reviewer to
add/improve them.


.. _pre-commit: https://pre-commit.com/
