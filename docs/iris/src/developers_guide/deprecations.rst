Deprecations
************

If you need to make a backwards-incompatible change to a public API
[#public-api]_ which has been included in a release, then you should
only make that change after a deprecation period. This deprecation
period must last at least six months or two public releases, whichever
results in the longer period of time.  Once the deprecation period has
expired the deprecated API should be removed/updated in the next
`major release <http://semver.org/>`_.

NB. Private APIs and public APIs defined in the :mod:`iris.experimental`
package are exempt from the deprecation rule - any change is allowed at
any time.


Adding a deprecation
====================


.. _removing-a-public-api:

Removing a public API
---------------------

The simplest form of deprecation occurs when you need to remove a public
API. Sometimes this is accompanied by the introduction of a new public
API.

 - Usage of the deprecated API must result in a brief warning.
 - Where possible, your deprecation warning should include advice on
   how to avoid the deprecated API. For example, you might reference a
   preferred API, or more detailed documentation elsewhere.
 - You must update the docstring for the deprecated API to include a
   Sphinx deprecation directive:

    :literal:`.. deprecated:: <VERSION>`

   As with the deprecation warning, you should include advice on how to
   avoid the deprecated API within the content of this directive. Feel
   free to include more detail on the change than in the deprecation
   warning.
 - You should check the documentation for references to the deprecated
   API and update them as appropriate.

Changing defaults
-----------------

When you need to change the default behaviour of a public API the
situation is slightly more complex and requires a two-step process.
The recommended solution is to use the :data:`iris.FUTURE` object.
The following points apply in addition to those for removing a public
API:

 - You should add a flag to :class:`iris.Future` which determines
   whether the default behaviour of the public API follows the current,
   deprecated behaviour or the new behaviour. This flag must default to
   the current behaviour.
 - You should include a reference to this :data:`iris.FUTURE` flag in your
   deprecation warning and the corresponding Sphinx directive.


Removing deprecations
=====================

When the time comes to make a new major release you should search the
code for any deprecated APIs which satisfy the six month/two release
minimum period described previously. This can easily be done by
searching for the Sphinx directives and/or warnings.

Removing a public API
---------------------

The deprecated API should be removed and any corresponding documentation
and/or example code should be removed/updated as appropriate.

Changing a default
------------------

 - You should update the default value of the relevant
   :class:`iris.Future` flag to select the new behaviour.
 - You should deprecate the relevant :class:`iris.Future` flag in the
   normal way, see :ref:`removing-a-public-api`.


.. rubric:: Footnotes

.. [#public-api] A name without a leading underscore.
