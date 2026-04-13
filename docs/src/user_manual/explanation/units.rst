.. explanation:: Units
   :tags: topic_data_model

   Read about how Iris objects such as Cubes and Coordinates are assigned scientific units.

.. include:: /common_links.inc

=====
Units
=====

.. todo:: cross reference in navigating_a_cube, why_iris, cube_maths, um_files_loading, glossary

A measure such as 'temperature' cannot just be described by a number - e.g.
273.15 - it must also be associated with a unit - e.g. 'Kelvin' - to give it
meaning. This is a core element of the :term:`CF Conventions`, and so is
fundamental to the design of Iris:

- All data model objects - e.g. :term:`Cube`, :term:`Coordinate`; anything
  based on :class:`~iris.common.mixin.CFVariableMixin` - have a
  :attr:`~iris.common.mixin.CFVariableMixin.units` attribute.
- Operations on Iris objects are 'recorded' in their units attribute. E.g.
  a :class:`~iris.cube.Cube` with units of ``m/s`` multiplied by a
  :class:`~iris.cube.Cube` with units of ``s`` will have units of ``m`` after
  the operation. Read more: :ref:`cube_maths_combining_units`.
- Operations between :class:`~iris.cube.Cube` objects - e.g. arithmetic or
  merging - are only permitted if the units are compatible. Read more:
  :doc:`/user_manual/section_indexes/metadata_arithmetic`,
  :ref:`merge_and_concat`.

In addition to the CF Conventions, the Iris ecosystem defines two further units:

- ``no-unit`` - the associated data is not suitable for describing with a unit.
- ``unknown`` - the unit describing the associated data cannot be determined.
  If a calculation is prevented because it would result in inappropriate units,
  it may be forced by setting the units of the original cubes to be
  ``"unknown"``.

The Units API
-------------

.. testsetup::

    from iris.cube import Cube
    from iris.experimental.units import USE_CFPINT
    my_cube = Cube([0, 1, 2])

Setting the :attr:`~iris.common.mixin.CFVariableMixin.units` on an object is
simple - you provide a string and Iris will parse and store it appropriately:

   >>> my_cube.units = 'm/s'
   >>> print(my_cube.units)
   m/s
   >>> print(repr(my_cube.units))
   Unit('m/s')

As well as a string, you can assign objects directly from the unit-handling
library and these will also be parsed. If you want to test this: the function
:func:`iris.common.units.make_unit` outputs the same object that would be
created when assigning an input to the
:attr:`~iris.common.mixin.CFVariableMixin.units` attribute.

You can choose which library you want Iris to use for unit parsing and operations:

.. list-table:: Choice of Cf-units or Pint in Iris
   :header-rows: 1
   :stub-columns: 1

   * - Library
     - Iris Class
     - Parent Class
     - Status @ ``2026-04-13``
   * - `Cf-units`_
     - :class:`iris.common.units.CfUnit`
     - :class:`cf_units.Unit`
     - Default
   * - `Pint`_
     - :class:`iris.common.units.PintUnit`
     - :class:`pint.Unit`
     - Experimental

You make this choice using the :data:`iris.experimental.units.USE_CFPINT` flag:

   >>> with USE_CFPINT.context():
   ...     my_cube.units = "m/s"
   ...     print(my_cube.units)
   m s-1
   ...     print(repr(my_cube.units))
   Unit('meter / second')

In both cases Iris internals ensure the unit is CF-compliant. CF-compliance is
standard behaviour for Cf-units; while the Pint case currently (``2026-04-13``)
uses the `Cfpint`_ library with some Iris-specific modifications. The intent is a
**seamless user experience**
regardless of the underlying library, hopefully allowing a
**gradual transition to `Pint`_**,
bringing Iris users closer to the wider scientific Python ecosystem.

To aid the above transition, any compatibility features have been marked as
deprecated, with advice on how to update your code. Example:
:attr:`iris.common.units.PintUnit.is_dimensionless`.

The Libraries Underneath
------------------------

The :term:`CF Conventions` officially define unit behaviour as follows:

- Reference time units: e.g. ``days since 2000-12-01``, behaviour
  described directly in the `CF Conventions`_ pages; section 4.4.
- All other units: behaviour provided by the `UDUNITS2`_ package, with a small
  number of modifications described in the `CF Conventions`_ pages; section 3.1.

Since reference time units are not based on existing software, the rules given
by the CF Conventions are implemented by the `Cftime`_ Python package.

A full software implementation of CF Conventions units must therefore combine:

- cftime
- UDUNITS2
- The specific UDUNITS2 modifications

The two most established packages for this are: `Cf-units`_ and `Cfunits`_.

Both of these packages have
struggled to find ways of combining Python with UDUNITS2, especially when it
comes to installation. These struggles have inspired attempts to implement
UDUNITS2 in Python or via the `Pint`_ Python package, all of which are experimental
at time of writing (``2026-04-13``).

Iris' hope for the future is for a mature and well maintained implementation of
**CF-compliant Pint**. The CF Conventions are the de facto standard for storing
atmospheric/oceanographic data, while Pint is the most widely accepted Python
package for units. Being Pint-based (rather than UDUNITS2-based), should
improve the Iris user experience:

- Better interoperability/collaboration with the wider scientific Python ecosystem.
- A better maintained units library - more features, more support.
- Simpler installation. Specifically: this would allow Iris to be fully
  installed via Pip (UDUNITS2 needs Conda for installation).

This is why we are future-proofing Iris to support both Cf-units and Pint, and
why we are working with international collaborators to establish a
CF-compliant Pint implementation.

.. _Cf-units: https://github.com/SciTools/cf-units
.. _Cfunits: https://github.com/NCAS-CMS/cfunits
.. _Pint: https://github.com/hgrecco/pint
.. _Cfpint: https://github.com/SciTools/cfpint
.. _UDUNITS2: https://www.unidata.ucar.edu/software/udunits
.. _Cftime: https://github.com/unidata/cftime
