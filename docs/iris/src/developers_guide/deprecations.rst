Deprecations
************

If you need to make a backwards-incompatible change to a public API
[#public-api]_ that has been included in a release, then you should
only make that change after a deprecation period. This deprecation
period must last at least six months or two public releases, whichever
results in the longer period of time.  Once the deprecation period has
expired the deprecated API should be removed/updated in the next
`major release <http://semver.org/>`_.


Adding a deprecation
====================

.. _removing-a-public-api:

Removing a public API
---------------------

The simplest form of deprecation occurs when you need to remove a public
API. The public API in question is deprecated for a period before it is
removed to allow time for user code to be updated. Sometimes the
deprecation is accompanied by the introduction of a new public API.

Under these circumstances the following points apply:

 - Using the deprecated API must result in a concise deprecation
   warning.
 - Where possible, your deprecation warning should include advice on
   how to avoid using the deprecated API. For example, you might
   reference a preferred API, or more detailed documentation elsewhere.
 - You must update the docstring for the deprecated API to include a
   Sphinx deprecation directive:

    :literal:`.. deprecated:: <VERSION>`

   where you should replace `<VERSION>` with the major and minor version
   of Iris in which this API is first deprecated. For example: `1.8`.

   As with the deprecation warning, you should include advice on how to
   avoid using the deprecated API within the content of this directive.
   Feel free to include more detail in the updated docstring than in the
   deprecation warning.
 - You should check the documentation for references to the deprecated
   API and update them as appropriate.

Changing a default
------------------

When you need to change the default behaviour of a public API the
situation is slightly more complex. The recommended solution is to use
the :data:`iris.FUTURE` object. The :data:`iris.FUTURE` object provides
boolean attributes that allow user code to control at run-time the
default behaviour of corresponding public APIs. When a boolean attribute
is set to `False` it causes the corresponding public API to use its
deprecated default behaviour. When a boolean attribute is set to `True`
it causes the corresponding public API to use its new default behaviour.

The following points apply in addition to those for removing a public
API:

 - You should add a new boolean attribute to :data:`iris.FUTURE` (by
   modifying :class:`iris.Future`) that controls the default behaviour
   of the public API that needs updating. The initial state of the new
   boolean attribute should be `False`. You should name the new boolean
   attribute to indicate that setting it to `True` will select the new
   default behaviour.
 - You should include a reference to this :data:`iris.FUTURE` flag in your
   deprecation warning and corresponding Sphinx deprecation directive.


Removing a deprecation
======================

When the time comes to make a new major release you should locate any
deprecated APIs within the code that satisfy the six month/two release
minimum period described previously. Locating deprecated APIs can easily
be done by searching for the Sphinx deprecation directives and/or
deprecation warnings.

Removing a public API
---------------------

The deprecated API should be removed and any corresponding documentation
and/or example code should be removed/updated as appropriate.

Changing a default
------------------

 - You should update the initial state of the relevant boolean attribute
   of :data:`iris.FUTURE` to `True`.
 - You should deprecate setting the relevant boolean attribute of
   :class:`iris.Future` in the same way as described in
   :ref:`removing-a-public-api`.


.. rubric:: Footnotes

.. [#public-api] A name without a leading underscore in any of its
   components, with the exception of the :mod:`iris.experimental` and
   :mod:`iris.tests` packages.

   Example public names are:
    - `iris.this.`
    - `iris.this.that`

   Example private names are:
    - `iris._this`
    - `iris.this._that`
    - `iris._this.that`
    - `iris._this._that`
    - `iris.experimental.something`
    - `iris.tests.get_data_path`
