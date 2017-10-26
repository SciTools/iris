.. _change_management:

Change Management in Iris from the User's perspective
*****************************************************

As Iris changes, user code will need revising from time to time to keep it
working, or to maintain best practice.  At the very least, you are advised to
review existing code to ensure it functions correctly with new releases.

Here, we define ways to make this as easy as possible.

.. include:: ../userguide/change_management_goals.txt


Key principles you can rely on
==============================

Iris code editions are published as defined version releases, with a given
major and minor version number in the version name, "major.minor.xxx",
as explained in the :ref:`releases section <iris_change_releases>` below.

    * Code that currently works should **still work**, and have the same
      results and effects, in any subsequent sub-release with the same major
      release number.

    * The only time we will make changes that can break existing code is at
      a **major release**.

    * At a major release, code that works **and emits no deprecation warnings**
      in the latest previous (minor) release should still work, and have the
      same results and effects.


**What can possibly go wrong ?**

If your code produces :ref:`deprecation warnings <iris_deprecations>`, then it
*could* behave differently, or fail, at the next major release.



User Actions : How you should respond to changes and releases
=============================================================

Checklist :

* when a new **testing or candidate version** is announced
    if convenient, test your working legacy code against it and report any problems.

* when a new **minor version is released**

    * review the 'Whats New' documentation to see if it introduces any
      deprecations that may affect you.
    * run your working legacy code and check for any deprecation warnings,
      indicating that modifications may be necessary at some point
    * when convenient :

      * review existing code for use of deprecated features
      * rewrite code to replace deprecated features

* when a new major version is **announced**
    ensure your code runs, without producing deprecation warnings, in the
    previous minor release

* when a new major version is **released**
    check for new deprecation warnings, as for a minor release


Details
=======

The Iris change process aims to minimise the negative effects of change, by
providing :

    * defined processes for release and change management
    * release versioning
    * backwards code compatibility through minor version releases
    * a way to ensure compatibility with a new major version release
    * deprecation notices and warnings to highlight all impending changes

Our practices are intended be compatible with the principles defined in the
`SemVer project <http://semver.org/>`_ .

Key concepts covered here:
    * :ref:`Release versions <iris_change_releases>`
    * :ref:`Backwards compatibility <iris_backward_compatibility>`
    * :ref:`Deprecation <iris_deprecations>`


.. _iris_backward_compatibility:

Backwards compatibility
-----------------------

"Backwards-compatible" changes are those that leave any existing valid API
usages unchanged (see :ref:`terminology <iris_api>` below).
Minor releases may only include backwards-compatible changes.

The following are examples of backward-compatible changes :

    * changes to documentation
    * adding to a module : new submodules, functions, classes or properties
    * adding to a class : new methods or properties
    * adding to a function or method : new **optional** arguments or keywords

The following are examples of **non-** backward-compatible changes :

    * removing (which includes *renaming*) any public module or submodule
    * removing any public component : a module, class, method, function or
        data object property of a public API component
    * removing any property of a public object
    * removing an argument or keyword from a method or function
    * adding a required argument to a method or function
    * removing a keyword (even one that has no effect)
    * changing the effect of *any* particular combination of arguments and/or
      keywords

Note that it is also possible to modify the behaviour of an existing usage by
making it depend on a newly-defined external control variable.  This is
effectively a change to the 'default behaviour' of a specific usage.  Although
this seems similar to adding a keyword, the cases where the new behaviour
operates and where it does not are not distinguishable by a different code
usage, which makes this somewhat dangerous.  We do use this type of change,
but any behaviour 'mode' controls of this sort are usually added as part of the
:class:`iris.Future` definition.
See :ref:`Usage of iris.FUTURE <iris_future_usage>`, below.


.. _iris_api:

Terminology : API, features, usages and behaviours
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The API is the components of the iris module and its submodules which are
"public" :  In Python, by convention, this normally means everything whose name
does not have a single leading underscore "_".
This applies to all public modules and their properties : submodules, contained
public classes, data and properties, functions and methods.
An exception is when a module or class defines an '__all__' property :  In
that case, the public aspects are just the ones listed there.

Note: these are standard Python conventions, not specific to Iris.
See: `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.

The Iris project considers all public API features as "supported", which means
that we will not change or remove them without deprecation, and will undertake
to fix any bugs discovered.
We do however make an important exception for the content of the 'experimental'
module :  APIs in :mod:`iris.experimental` module are published for initial
evaluation and feedback, and can  be revised or removed without warning at a
subsequent release.

A "feature" of the API includes public objects as above, but may also be used
more loosely to indicate a class or mode of behaviour, for example when a
keyword has a specific value, like "interpolate(mode='linear')".

A "usage" is any code referring to public API elements, for example :

        * `print(iris.thing)`
        * `iris.submodule.call(arg1)`
        * `iris.module.call(arg1, arg2, *more_args)`
        * `iris.module.call(arg1, arg2, control=3)`
        * `x = iris.module.class(arg, key=4)`

A "behaviour" is whatever Iris does when you invoke a particular API usage,
encompassing both returned values and any side effects such as code state
changes or data written to files.

As the above code examples are all public feature usages, they should
therefore continue to work, with the same behaviours, at least until the next
**major** version release.

.. Note::
    What is the "same" behaviour, for backwards-compatibility purposes ?

    Unfortunately, the guarantee to preserve "what Iris does" within a major
    version cannot ensure *totally* identical and repeatable behaviour for any
    possible usage, because this can also depend on the exact installed
    versions of all dependencies (i.e. the other Python modules and system
    libraries that Iris uses).

    See :ref:`iris_dependency_versions`.

    Minor-release code changes are backwards-compatible, meaning that they must
    result in "the same" actions from Iris.
    Ultimately, however, this is only a statement of intent, as we need freedom
    to modify code with "no change" effects, yet *any* change to Iris code or
    dependencies could alter total behaviour in some respects :
    For instance, it could take more or less time or memory, produce results in
    a different sequence, or perform a calculation slightly differently (with
    possible small differences in floating point results).

    As behaviour can only be tested in specific ways on a specific
    installation, so any given user installation may experience changes in
    behaviour, though hopefully always slight, with a minor release change.
    This applies to changes in the Iris minor release version, or a different
    version of some dependency; or other changes to the operating system
    environment.


.. _iris_change_releases:

Releases and Versions
---------------------


Iris releases have a unique identifying version string, in the form
"<major>.<minor>.<micro><extension>", available to code as
:data:`iris.__version__` .

This contains major and minor release numbers.  The numbering and meaning of
these are defined, following the `SemVer project <http://semver.org/>`_.

The essential aspects of the "<major>.<minor>.<micro><extension>" arrangement
are :

    * "<major>", "<minor>" and "<micro>" are all integers, thus version
      2.12 is later than 2.2 (i.e. it is "two point twelve", not "two point one
      two").

    * "<major>.<minor>" denote the software release version.

    * A non-zero "<micro>" denotes a bugfix version, thus a release "X.Y.0" may
      be followed by "X.Y.1", "X.Y.2" etc, which *only* differ by containing
      bugfixes.  Any bugfix release supercedes its predecessors, and does not
      change any (valid) APIs or behaviour :  hence, it is always advised to
      replace a given version with its latest bugfix successor, and there
      should be no reason not to.

    * "<extension>" is blank for formal releases.  It used to indicate
      provisional software for testing :  The version string in a development
      code branch is always labelled "-DEV", and release candidates for testing
      during the release process are labelled "-rc1", "-rc2" etc.
      For development code, the version number is that of the *next* release,
      which this code version is progressing towards, e.g. "1.2-DEV" for all
      code branches since the 1.1 release and intended for release in "1.2".

.. note::
    Our use of "-<extension>" is typical, but does not follow strict SemVer
    principles.

The code for a specific release is identified by a git tag which is the version
string : see
:ref:`Developer's Guide section on releases <iris_development_releases>`.


Major and Minor Releases
^^^^^^^^^^^^^^^^^^^^^^^^

The term "release" refers both to a specific state of the Iris code, which we
have assigned a given version string, *and* the act of defining it
(i.e. we "release a release").

According to `SemVer <http://semver.org/>`_ principles, changes that alter the
behaviour of existing code can only be made at a **major** release, i.e. when
"X.0" is released following the last previous "(X-1).Y.Z".

*Minor* releases, by contrast, consist of bugfixes, new features, and
deprecations :  Any valid exisiting code should be unaffected by these, so it
will still run with the same results.

At a major release, only **deprecated** behaviours and APIs can be changed or
removed.


.. _iris_deprecations:

Deprecations
------------

A deprecation is issued when we decide that an *existing* feature needs to be
removed or modified :  We add notices to the documentation, and issue a runtime
"Deprecation Warning" whenever the feature is used.

For a wider perspective, see : `<https://en.wikipedia.org/wiki/Deprecation>`_.  
For the developer view of this, see
:ref:`Developer's Guide section on deprecations <iris_development_deprecations>`.

Deprecation manages incompatible changes in a strictly controlled way.
This allows APIs to evolve to the most effective form, even when that means
that existing code could behave differently or fail :  This is important
because the freedom to remove features helps prevent the API becoming
progressively cluttered, and modifying existing behaviours allows us to use
the most natural code syntax for the most used features.

We can only remove features or change behaviours at a major release.  Thus, we
first deprecate the feature in a minor release, to provide adequate warning
that existing code may need to be modified.

When we make a release that introduces a deprecation :

    * a deprecation notice appears in the
      :ref:`What's New section <iris_whatsnew>`
    * deprecation notes are included in all relevant parts of the :ref:`reference
      documentation <iris>`
    * a runtime warning is produced when the old feature is used or triggered

In most cases, we also provide detailed advice in the documentation and/or
warning messages on how to replace existing usage with a 'new' way of doing
things.
In all cases, we must provide a transitional period where both old and new
features are available :

    * the 'old' style works as it did before
    * any usage of the 'old' features will emit a
      :class:`warnings.WarningMessage` message, noting that the feature is
      deprecated and what to use instead
    * the 'new' style can be adopted as soon as convenient

This is to warn users :

    * not to use the deprecated features in any new code, *and*
    * eventually to rewrite old code to use the newer or better alternatives


Deprecated features support through the Release cycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The whole point of a deprecation is that the feature continues to work, but
with a warning, for some time before an unavoidable change occurs.
When a version that introduces a deprecation is released, the effects are as
follows:

    * code that may be affected by the proposed change will result in
      deprecation warnings
    * code that currently works will, however, continue to work unchanged, at
      least until the next major release
    * you can avoid all deprecation warnings by suitable changes to your code
    * code which uses no deprecated features, and thus produces no deprecation
      warnings, will continue to work unchanged even at a **major** release
    * code that generates deprecation warnings may cease to work at the next
      **major** release.


.. _iris_future_usage:

Future options, `iris.FUTURE`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A special approach is needed where the replacement behaviour is not controlled
by a distinct API usage.

When we extend an API, or add a new feature, we usually add a new method or
keyword. In those cases, code using the new feature is clearly distinct from
any previous (valid) usage, so this is relatively simple to manage.
However, sometimes we really need to change the *way* an API works, without
modifying or extending (i.e. complicating) the existing user interface.
In that case, existing user code might sometimes have *different* behaviour
with the new release, which we obviously need to avoid.

**For example :**

    We might decide there is a more useful way of loading cubes from files of a
    particular input data format.

        * the user code usage is simply by calls to "iris.load"
        * the change is not a bugfix, as the old way isn't actually "wrong"
        * we don't want to add an extra keyword into all the relevant calls
        * we don't see a longterm future for the existing behaviour :  we
          expect everyone to adopt the new interpretation, eventually

For changes of this sort, the release will define a new boolean property of the
:data:`iris.FUTURE` object, as a control to select between the 'old' and 'new'
behaviours, with values False='old' and True='new'.
See :data:`iris.Future` for examples.

In these cases, as any "deprecated usage" is not clearly distinguishable in the
form of the user code, it is **especially** important to take note of any
deprecation messages appearing when legacy code runs.


**Sequence of changes to `iris.FUTURE`**

To allow user code to avoid unexpected any behavioural changes, the
:data:`iris.Future` controls follow a special management cycle, as follows
(see also the relevant :ref:`Developer Guide section <iris_developer_future>`):

At (minor) release "<X>.<Y>..":
    * Changes to API:
        * the new behaviour is made available, alongside the old one

        * a new future option `iris.FUTURE.<new_enable>` is provided to switch
          between them.

        * the new option defaults to `iris.FUTURE.<new_enable>=False`, meaning
          the 'old' behaviour is the default.

        * when any relevant API call is made that invokes the old behaviour, a
          deprecation warning is emitted.

    * User actions:

        * If your code encounters the new deprecation warning, you should try
          enabling the new control option, and make any necessary rewrites to
          make it work.  This will stop the deprecation warning appearing.

        * If you encounter problems making your code work with the new
          behaviour, and don't have time to fix them, you should make your
          code explicitly turn *off* the option for now, i.e. ::
          `iris.FUTURE.<new_enable> = False`.
          This locks you into the old behaviour, but your code  will continue
          to work, even beyond the next major release when the default
          behaviour will change (see on).

At (major) release "<X+1>.0...":
    * Changes to API:
        * the control default is changed to `iris.FUTURE.<new_enable>=True`

        * the control property is *itself* deprecated, so that assigning to it
          now results in a deprecation warning.

        * when any affected API call is made, a deprecation warning is (still)
          emitted, if the old behaviour is in force.  The "old" behaviour is,
          however, still available and functional.

    * User actions:

        * If your code is already using the "new" behaviour, it will now work
          without needing to set the Future option.  *You should remove* the
          code which enables the option, as this will now emit a deprecation
          message.  In the *next* major release, this would cause an error.

        * If your code is explicitly turning the option off, it will continue
          to work in the same way at this point, but obviously time is
          runnning out.

        * If your code is still using the old behaviour and *not* setting the
          control option at all, its behaviour might now have changed
          unexpectedly and you should review this.

At (major) release "<X+2>...":
    * Changes to API:
        * the control property is removed
        * the "old" behaviour is removed



.. _iris_dependency_versions:

Versions of Installed Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dependencies of Iris (required and optional) are defined in
:ref:`installing_iris`.

Iris does not specify exact required versions for its dependencies, but it may
specify a minimum acceptable version number.  Iris is normally expected to be
compatible with *any* version up to the latest current release.

When a new release of a dependency is found to cause problems, Iris may define
the supported version more precisely, but this would be a temporary fix which
should be removed in a later release.
