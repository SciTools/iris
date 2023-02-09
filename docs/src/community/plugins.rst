.. _namespace package: https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

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
