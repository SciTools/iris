.. include:: ../common_links.inc
.. _namespace package: https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

.. todo:
    consider scientific-python.org
    consider scientific-python.org/specs/

Iris in the Community
=====================

Iris aims to be a valuable member of the open source scientific Python
community.

We listen out for developments in our dependencies and neighbouring projects,
and we reach out to them when we can solve problems together; please feel free
to reach out to us!

We are aware of our place in the user's wider 'toolbox' - offering unique
functionality and interoperating smoothly with other packages.

We welcome contributions from all; whether that's an opinion, a 1-line
clarification, or a whole new feature 🙂

Quick Links
-----------

* `GitHub Discussions`_
* :ref:`Getting involved<development_where_to_start>`
* `Twitter <https://twitter.com/scitools_iris>`_

Interoperability
----------------

There's a big choice of Python tools out there! Each one has strengths and
weaknesses in different areas, so we don't want to force a single choice for your
whole workflow - we'd much rather make it easy for you to choose the right tool
for the moment, switching whenever you need. Below are our ongoing efforts at
smoother interoperability:

.. not using toctree due to combination of child pages and cross-references.

* The :mod:`iris.pandas` module
* :doc:`iris_xarray`

.. toctree::
   :maxdepth: 1
   :hidden:

   iris_xarray

Plugins
-------

Iris supports *plugins* under the `iris.plugins` `namespace package`_.  This
allows packages that extend Iris' functionality to be developed and maintained
independently, while still being installed into `iris.plugins` instead of a
separate package.  For example, a plugin may provide loaders or savers for
additional file formats, or alternative visualisation methods.

Once a plugin is installed, it can be used either via the :func:`iris.use_plugin`
function, or by importing it directly:

.. code-block:: python

    import iris

    iris.use_plugin("my_plugin")
    # OR
    import iris.plugins.my_plugin
