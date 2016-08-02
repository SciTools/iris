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
     are simply waiting until an update becomes unavoidable :

     Often, that will eventually occur when you need to update some other
     software component, for a completely unconnected reason.


Goals of Change Management
--------------------------

When you do upgrade Iris to a new version, you could potentially find
that you need to rewrite your legacy code, simply to keep it working.

.. include:: change_management_goals.txt

To take advantage of this, you should read the basic change management
recommendations laid out in :ref:`change_management`.
