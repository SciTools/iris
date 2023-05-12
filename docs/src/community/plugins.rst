.. _namespace package: https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

.. _community_plugins:

Plugins
=======

Iris supports **plugins** under the ``iris.plugins`` `namespace package`_.
This allows packages that extend Iris' functionality to be developed and
maintained independently, while still being installed into ``iris.plugins``
instead of a separate package.  For example, a plugin may provide loaders or
savers for additional file formats, or alternative visualisation methods.


Using plugins
-------------

Once a plugin is installed, it can be used either via the
:func:`iris.use_plugin` function, or by importing it directly:

.. code-block:: python

    import iris

    iris.use_plugin("my_plugin")
    # OR
    import iris.plugins.my_plugin


Creating plugins
----------------

The choice of a `namespace package`_ makes writing a plugin relatively
straightforward: it simply needs to appear as a folder within ``iris/plugins``,
then can be distributed in the same way as any other package.  An example
repository layout:

.. code-block:: text

    + lib
      + iris
        + plugins
          + my_plugin
            - __init__.py
            - (more code...)
    - README.md
    - pyproject.toml
    - setup.cfg
    - (other project files...)

In particular, note that there must **not** be any ``__init__.py`` files at
higher levels than the plugin itself.

The package name - how it is referred to by PyPI/conda, specified by
``metadata.name`` in ``setup.cfg`` - is recommended to include both "iris" and
the plugin name.  Continuing this example, its ``setup.cfg`` should include, at
minimum:

.. code-block:: ini

    [metadata]
    name = iris-my-plugin

    [options]
    packages = find_namespace:

    [options.packages.find]
    where = lib
