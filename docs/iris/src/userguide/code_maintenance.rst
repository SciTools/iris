Code Maintenance
================

From a user point of view "code maintenance" means ensuring that your existing
working code stays working, in the face of changes to Iris.


Stability and Change
---------------------

In practice, as Iris develops, most users will want to periodically upgrade
their installed version to access new features or at least bugfixes.

This is obvious if you are still developing other code that uses Iris, or using
code from other sources.  
However, even if you have only legacy code that remains untouched, some code
maintenance effort is probably still necessary :

   * On the one hand, *in principle*, working code will go on working, as long
     as you don't change anything else.

   * However, such "version statis" can easily become a growing burden, if you
     are simply waiting until an update becomes unavoidable :  Often, that will
     eventually occur when you need to update some other software component,
     for some completely unconnected reason.


Principles of Change Management
-------------------------------

When you upgrade software to a new version, you often find that you need to
rewrite your legacy code, simply to keep it working.

In Iris, however, we aim to reduce code maintenance problems to an absolute
minimum by following defined change management rules.
These ensure that, *within a major release number* :

    * you can be confident that your code will still work with subsequent minor
      releases

    * you will be aware of future incompatibility problems in advance

    * you can defer making code compatibility changes for some time, until it
      suits you

The above applies to minor version upgrades : e.g. code that works with version
"1.4.2" should still work with a subsequent minor release such as "1.5.0" or
"1.7.2".

A *major* release however, e.g. "v2.0.0" or "v3.0.0", can include more
significant changes, including so-called "breaking" changes:  This means that
existing code may need to be modified to make it work with the new version.

Since breaking change can only occur at major releases, these are the *only*
times we can alter or remove existing behaviours (even deprecated
ones).  This is what a major release is for : it enables the removal and
replacement of old features.

Of course, even at a major release, we do still aim to keep breaking changes to
a minimum.
