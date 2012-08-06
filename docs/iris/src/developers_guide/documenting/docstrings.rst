================
 Docstrings
================


Guiding principle: Every public object in the Iris package should have an appropriate docstring.

This document has been influenced by the following PEP's,
   * Attribute Docstrings `PEP-224 <http://www.python.org/dev/peps/pep-0224/>`_ 
   * Docstring Conventions `PEP-257 <http://www.python.org/dev/peps/pep-0257/>`_


For consistency, always use ``"""triple double quotes"""`` around docstrings. Use ``r"""raw triple double quotes"""`` if you use any backslashes in your docstrings. For Unicode docstrings, use ``u"""Unicode triple-quoted string"""``.

All docstrings should be written in rST (reStructuredText) markup; an rST guide follows this page.

There are two forms of docstrings: **single-line** and **multi-line** docstrings.


Single-line docstrings
======================
The single line docstring of an object must state the *purpose* of that object, known as the *purpose section*. This terse overview must be on one line and ideally no longer than 90 characters.


Multi-line docstrings
=====================
Multi-line docstrings must consist of at least a purpose section akin to the single-line docstring, followed by a blank line and then any other content, as described below. The entire docstring should be indented to the same level as the quotes at the docstring's first line.


Description
-----------
The multi-line docstring  *description section* should expand on what was stated in the one line *purpose section*. The description section should try not to document *argument* and *keyword argument* details. Such information should be documented in the following *arguments and keywords section*.


Sample multi-line docstring
---------------------------
Here is a simple example of a standard dosctring:

.. literalinclude:: docstrings_sample_routine.py

This would be rendered as:

   .. currentmodule:: documenting.docstrings_sample_routine
   
   .. automodule:: documenting.docstrings_sample_routine
      :members:
      :undoc-members:

Additionally, a summary can be extracted automatically, which would result in:

   .. autosummary::

      documenting.docstrings_sample_routine.sample_routine


Documenting classes
===================
The class constructor should be documented in the docstring for its ``__init__`` or ``__new__`` method. Methods should be documented by their own docstring, not in the class header itself.

If a class subclasses another class and its behavior is mostly inherited from that class, its docstring should mention this and summarise the differences. Use the verb "override" to indicate that a subclass method replaces a superclass method and does not call the superclass method; use the verb "extend" to indicate that a subclass method calls the superclass method (in addition to its own behavior).


Attribute and Property docstrings
---------------------------------
Here is a simple example of a class containing an attribute docstring and a property docstring:

.. literalinclude:: docstrings_attribute.py

This would be rendered as:

   .. currentmodule:: documenting.docstrings_attribute

   .. automodule:: documenting.docstrings_attribute
      :members:
      :undoc-members:

.. note:: The purpose section of the property docstring **must** state whether the property is read-only.
