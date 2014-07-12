# (C) British Crown Copyright 2013 - 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.

"""
Ugrid functions.

"""
import iris
from pyugrid.ugrid import UGrid


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

    cube = iris.load_cube(location, name)
    ug = UGrid.from_ncfile(location)
    cube.mesh = ug
    cube.mesh_dimension = 1  # {0:time, 1:node}
    return cube
