.. _iris_code_format:

Code Formatting
***************

Iris uses `black <https://black.readthedocs.io/en/stable/>`_ formatting and `flake8 <https://flake8.pycqa.org/en/stable/>`_
linting to enforce a consistent code format throughout the project. ``black`` and ``flake8`` can easily be installed
into your development environment with ``pip``::

  $ pip install black flake8

and then manually run from the root of the Iris repository::

  $ black .
  $ flake8 .

Alternatively, ``black`` and ``flake8`` may be installed using ``conda`` instead. Note that, ``black`` also offers
`editor integration <https://black.readthedocs.io/en/stable/editor_integration.html#editor-integration>`_.

As a developer workflow, we instead recommend using `pre-commit <https://pre-commit.com/>`_ to automatically run ``black`` and ``flake8``
whenever you perform a ``git commit``. Please install ``pre-commit`` in your development environment as follows::

  $ pip install pre-commit

then run::

  $ pre-commit install

from the root of the Iris repository to install the Iris ``pre-commit`` git hooks defined in our ``.pre-commit-config.yaml``
file. Note that, the ``pre-commit`` hooks will download and install the appropriate packages as necessary.

Upon performing a ``git commit``, your code will now be automatically formatted according to our ``black`` configuration defined in the
``pyproject.toml`` file, and linted using our ``flake8`` configuration defined in the ``.flake8`` file.

Note that, ``pre-commit`` allows you to `disable hooks <https://pre-commit.com/#temporarily-disabling-hooks>`_
temporarily, if so required.
