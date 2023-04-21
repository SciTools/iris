.. include:: ../common_links.inc

|iris_version| |build_date| [unreleased]
****************************************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. dropdown:: |iris_version| Release Highlights
   :color: primary
   :icon: info
   :animate: fade-in
   :open:

   The highlights for this major/minor release of Iris include:

   * N/A

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. N/A


✨ Features
===========

#. `@pp-mo`_ and  `@lbdreyer`_ supported delayed saving of lazy data, when writing to
   the netCDF file format.  See : :ref:`delayed netCDF saves <delayed_netcdf_save>`.
   Also with significant input from `@fnattino`_.
   (:pull:`5191`)


🐛 Bugs Fixed
=============

#. N/A


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. N/A


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. `@rcomer`_ and `@bjlittle`_ (reviewer) added testing support for python
   3.11. (:pull:`5226`)

#. `@rcomer`_ dropped support for python 3.8, in accordance with the NEP29_
   recommendations (:pull:`5226`) 


📚 Documentation
================

#. `@tkknight`_ migrated to `sphinx-design`_ over the legacy `sphinx-panels`_.
   (:pull:`5127`)

#. `@tkknight`_ updated the ``make`` target for ``help`` and added
   ``livehtml`` to auto generate the documentation when changes are detected
   during development. (:pull:`5258`)


💼 Internal
===========

#. `@bjlittle`_ added the `codespell`_ `pre-commit`_ ``git-hook`` to automate
   spell checking within the code-base. (:pull:`5186`)

#. `@bjlittle`_ and `@trexfeathers`_ (reviewer) added a `check-manifest`_
   GitHub Action and `pre-commit`_ ``git-hook`` to automate verification
   of assets bundled within a ``sdist`` and binary ``wheel`` of our
   `scitools-iris`_ PyPI package. (:pull:`5259`)

#. `@rcomer`_ removed a now redundant copying workaround from Resolve testing.
   (:pull:`5267`)

#. `@bjlittle`_ and `@trexfeathers`_ (reviewer) migrated ``setup.cfg`` to
   ``pyproject.toml``, as motivated by `PEP-0621`_. (:pull:`5262`)

#. `@bjlittle`_ adopted `pypa/build`_ recommended best practice to build a
   binary ``wheel`` from the ``sdist``. (:pull:`5266`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:

.. _@fnattino: https://github.com/fnattino


.. comment
    Whatsnew resources in alphabetical order:

.. _sphinx-panels: https://github.com/executablebooks/sphinx-panels
.. _sphinx-design: https://github.com/executablebooks/sphinx-design
.. _check-manifest: https://github.com/mgedmin/check-manifest
.. _PEP-0621: https://peps.python.org/pep-0621/
.. _pypa/build: https://pypa-build.readthedocs.io/en/stable/
.. _NEP29: https://numpy.org/neps/nep-0029-deprecation_policy.html
