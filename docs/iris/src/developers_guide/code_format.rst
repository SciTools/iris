.. _iris_code_format:

Code formatting
***************

To enforce a consistent code format throughout Iris, we recommend using `pre-commit <https://pre-commit.com/>`_ to run
`black <https://black.readthedocs.io/en/stable/>`_ formatting and `flake8 <https://flake8.pycqa.org/en/stable/>`_
linting automatically prior to each ``git commit``.

Please install ``pre-commit`` in your development environment using ``pip``::

    $ pip install pre-commit

or alternatively using ``conda``::

    $ conda install -c conda-forge pre-commit

Then from the root directory of Iris run::

    $ pre-commit install

to install the ``pre-commit`` git hooks defined in our ``.pre-commit-config.yaml`` file.

Upon performing a ``git commit``, your code will now be automatically formatted to the ``black`` configuration defined
in our ``pyproject.toml`` file, and linted according to our ``.flake8`` configuration file. Note that, ``pre-commit``
will automatically download and install the necessary packages for each ``.pre-commit-config.yaml`` git hook.

Additionally, you may wish to enable ``black`` for your preferred `editor/IDE <https://black.readthedocs.io/en/stable/editor_integration.html#editor-integration>`_.
