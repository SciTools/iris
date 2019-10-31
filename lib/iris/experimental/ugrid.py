# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Ugrid functions.

"""

import iris


def ugrid(location, name):
    """
    Create a cube from an unstructured grid.

    Args:

    * location:
        A string whose value represents the path to a file or
        URL to an OpenDAP resource conforming to the
        Unstructured Grid Metadata Conventions for Scientific Datasets
        https://github.com/ugrid-conventions/ugrid-conventions

    * name:
        A string whose value represents a cube loading constraint of
        first the standard name if found, then the long name if found,
        then the variable name if found, before falling back to
        the value of the default which itself defaults to "unknown"

    Returns:
        An instance of :class:`iris.cube.Cube` decorated with
        an instance of :class:`pyugrid.ugrid.Ugrid`
        bound to an attribute of the cube called "mesh"

    """
    # Lazy import so we can build the docs with no pyugrid.
    import pyugrid

    cube = iris.load_cube(location, name)
    ug = pyugrid.ugrid.UGrid.from_ncfile(location)
    cube.mesh = ug
    cube.mesh_dimension = 1  # {0:time, 1:node}
    return cube
