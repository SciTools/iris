.. _docstrings:

==========
Docstrings
==========

Every public object in the Iris package should have an appropriate docstring.
This is important as the docstrings are used by developers to understand
the code and may be read directly in the source or via the :ref:`Iris`.

This document has been influenced by the following PEP's,

   * Attribute Docstrings :pep:`224`
   * Docstring Conventions :pep:`257`

For consistency always use:

* ``"""triple double quotes"""`` around docstrings.
* ``r"""raw triple double quotes"""`` if you use any backslashes in your
  docstrings.
* ``u"""Unicode triple-quoted string"""`` for Unicode docstrings

All docstrings should be written in reST (reStructuredText) markup.  See the
:ref:`reST_quick_start` for more detail.

There are two forms of docstrings: **single-line** and **multi-line**
docstrings.


Single-Line Docstrings
======================

The single line docstring of an object must state the **purpose** of that
object, known as the **purpose section**. This terse overview must be on one
line and ideally no longer than 80 characters.


Multi-Line Docstrings
=====================

Multi-line docstrings must consist of at least a purpose section akin to the
single-line docstring, followed by a blank line and then any other content, as
described below. The entire docstring should be indented to the same level as
the quotes at the docstring's first line.


Description
-----------

The multi-line docstring  *description section* should expand on what was
stated in the one line *purpose section*. The description section should try
not to document *argument* and *keyword argument* details. Such information
should be documented in the following *arguments and keywords section*.


Sample Multi-Line Docstring
---------------------------

Here is a simple example of a standard docstring:

.. literalinclude:: docstrings_sample_routine.py

This would be rendered as:

   .. currentmodule:: documenting.docstrings_sample_routine

   .. automodule:: documenting.docstrings_sample_routine
      :members:
      :undoc-members:

Additionally, a summary can be extracted automatically, which would result in:

   .. autosummary::

      documenting.docstrings_sample_routine.sample_routine


Documenting Classes
===================

The class constructor should be documented in the docstring for its
``__init__`` or ``__new__`` method. Methods should be documented by their own
docstring, not in the class header itself.

If a class subclasses another class and its behaviour is mostly inherited from
that class, its docstring should mention this and summarise the differences.
Use the verb "override" to indicate that a subclass method replaces a
superclass method and does not call the superclass method; use the verb
"extend" to indicate that a subclass method calls the superclass method
(in addition to its own behaviour).


Attribute and Property Docstrings
---------------------------------

Here is a simple example of a class containing an attribute docstring and a
property docstring:

.. literalinclude:: docstrings_attribute.py

This would be rendered as:

   .. currentmodule:: documenting.docstrings_attribute

   .. automodule:: documenting.docstrings_attribute
      :members:
      :undoc-members:

.. note:: The purpose section of the property docstring **must** state whether
          the property is read-only.
