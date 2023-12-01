.. include:: ../common_links.inc

.. _phrasebook:

Package Phrasebook
===================

There are a number of similar packages to Iris, and a lot of these have their own
terminology for similar things. Whether you're coming or going, we hope this might
be a helpful guide to these differences!
Definitions for each can be found in :ref:`glossary`. See also
`Xarray terminology <https://docs.xarray.dev/en/stable/user-guide/terminology.html>`_.

.. list-table:: Phrasebook
   :widths: 25 25 25 50
   :header-rows: 1

   * - Iris
     - Xarray
     - Example
     - Notes
   * - Non-Lazy
     - Eager
     -
     - Used to relate to functions, rather than the data.
   * - Cube
     - DataArray
     -
     -
   * - CubeList
     - Dataset
     -
     - Though similar, a CubeList is a simpler object, and is
       not a perfect comparison to a Dataset
   * - Merge/ Concatenate
     - Concatenate
     - `Xarray concatenate <https://docs.xarray.dev/en/stable/user-guide/combining.html#concatenate>`_
     - Xarray's concatenate has the capability to largely do what both
       Iris merge and Iris concatenate do. However, this is not a perfect comparison,
       please see the link for more information.
   * -
     - Merge
     - `Xarray merge <https://docs.xarray.dev/en/stable/user-guide/combining.html#merge>`_
     - Xarray's Merge function doesn't map neatly map to any Iris feature.
       Please see the link for more information.
   * - Scalar Coordinate
     -
     -
     - Iris makes a distinction between scalar coordinates and non-scalar coordinates,
       whereas xarray documentation makes a distinction between scalar and non-scalar *data*.
       It is possible to make coordinates with scalar data in both Iris and xarray
       but only Iris will label such coordinates.
   * - AuxCoord
     - Non-Dimensional Coordinate
     -
     - Coordinates in Iris and xarray are categorised using different rules,
       and so are not a one-to-one match.
   * - DimCoord
     - Dimension Coordinate
     -
     - Coordinates in Iris and xarray are categorised using different rules,
       and so are not a one-to-one match.

----

`To top <phrasebook_>`_