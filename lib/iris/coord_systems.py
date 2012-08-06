# (C) British Crown Copyright 2010 - 2012, Met Office
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
Definitions of coordinate systems.

"""

from __future__ import division
from abc import ABCMeta

import numpy

import iris.cube
import iris.exceptions
import iris.util


# Define Horizontal coordinate type constants
CARTESIAN_CS = 'cartesian'
SPHERICAL_CS = 'spherical'
# maintain a tuple of valid CS types
_VALID_CS_TYPES = (CARTESIAN_CS, SPHERICAL_CS)


USE_OLD_XML = True


class SpheroidDatum(iris.util._OrderedHashable):
    """Defines the shape of the Earth."""

    # Declare the attribute names relevant to the _OrderedHashable behaviour.
    _names = ('label', 'semi_major_axis', 'semi_minor_axis', 'flattening', 'units')

    label = None
    """The name of this spheroid definition."""

    semi_major_axis = None
    """The length of the semi-major axis, :math:`a`."""

    semi_minor_axis = None
    """The length of the semi-minor axis, :math:`b`."""

    flattening = None
    """The flattening, :math:`f`, or ellipticity. Defined as :math:`f = 1-\\frac{b}{a}`."""

    units = None
    """The unit of measure for the axes."""

    def __init__(self, label='undefined spheroid', semi_major_axis=None, semi_minor_axis=None,
                        flattening=None, units='no unit'):
        """
        If all three of semi_major_axis, semi_minor_axis, and flattening are None then
        it defaults to a perfect sphere using the radius 6371229m.

        Otherwise, at least two of semi_major_axis, semi_minor_axis, and flattening
        must be given.

        """
        #if radius/flattening are not specified, use defaults.
        if (semi_major_axis is None) and (semi_minor_axis is None) and (flattening is None):
            #Use the UM radius from  http://fcm2/projects/UM/browser/UM/trunk/src/constants/earth_constants_mod.F90
            semi_major_axis = 6371229.0
            semi_minor_axis = 6371229.0
            flattening = 0.0
            units = iris.unit.Unit('m')

        #calculate the missing element (if any) from the major/minor/flattening triplet
        else:
            if semi_major_axis is None:
                if semi_minor_axis is None or flattening is None:
                    raise ValueError("Must have at least two of the major/minor/flattening triplet")
                semi_major_axis = semi_minor_axis / (1.0-flattening)

            elif semi_minor_axis is None:
                if semi_major_axis is None or flattening is None:
                    raise ValueError("Must have at least two of the major/minor/flattening triplet")
                semi_minor_axis = (1.0-flattening) * semi_major_axis

            elif flattening is None:
                if semi_major_axis is None or semi_minor_axis is None:
                    raise ValueError("Must have at least two of the major/minor/flattening triplet")
                flattening = 1.0 - (semi_minor_axis/semi_major_axis)

        self._init(label, semi_major_axis, semi_minor_axis, flattening, units)

    def is_spherical(self):
        """Returns whether this datum describes a perfect sphere."""
        return self.flattening == 0.0


class PrimeMeridian(iris.util._OrderedHashable):
    """Defines the origin of the coordinate system."""

    # Declare the attribute names relevant to the _OrderedHashable behaviour.
    _names = ('label', 'value')

    label = None
    """The name of the specific location which defines the reference point."""

    value = None
    """The longitude of the reference point."""

    def __init__(self, label="Greenwich", value=0.0):
        """ """
        self._init(label, value)


class GeoPosition(iris.util._OrderedHashable):
    """Defines a geographic coordinate latitude/longitude pair."""

    # Declare the attribute names relevant to the _OrderedHashable behaviour.
    _names = ('latitude', 'longitude')

    latitude = None
    """The latitude of the position in degrees."""

    longitude = None
    """The longitude of the position in degrees."""


class CoordSystem(object):
    """Abstract base class for coordinate systems.

    A Coord holds an optional CoordSystem, which can be used to indicate
    several Coords are defined to be 'in the same system'.
    E.g lat and lon coords will hold a shared or identical LatLonCS.
    """
    __metaclass__ = ABCMeta

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__dict__ == other.__dict__
        
    def __ne__(self, other):
        # Must supply __ne__, Python does not defer to __eq__ for negative equality
        return not (self == other)

    def assert_valid(self):
        """Check the CS is in a valid state (else raises error)."""
        pass

    def xml_element(self, doc):
        """Default behaviour for coord systems."""
        xml_element_name = type(self).__name__
        # lower case the first char
        xml_element_name = xml_element_name.replace(xml_element_name[0], xml_element_name[0].lower(), 1)
        
        coord_system_xml_element = doc.createElement(xml_element_name)
        
        attrs = []
        for k, v in self.__dict__.iteritems():
            if isinstance(v, iris.cube.Cube):
                attrs.append([k, 'defined'])
            else:
                if USE_OLD_XML:    
                    v = str(v).replace("units", "unit")
                    attrs.append([k, v])
                else:
                    attrs.append([k, v])

        attrs.sort(key=lambda attr: attr[0])     

        for name, value in attrs: 
            coord_system_xml_element.setAttribute(name, str(value))
        
        return coord_system_xml_element
    
    
class HorizontalCS(CoordSystem):
    """Abstract CoordSystem for holding horizontal grid information."""

    def __init__(self, datum):
        """ """
        CoordSystem.__init__(self)

        self.datum = datum
        self.cs_type = CARTESIAN_CS

    def __repr__(self):
        return "HorizontalCS(%r, %r)" % (self.datum, self.cs_type)

    def assert_valid(self):
        if self.cs_type not in _VALID_CS_TYPES:
            raise iris.exceptions.InvalidCubeError('"%s" is not a valid coordinate system type.' % self.cs_type)
        CoordSystem.assert_valid(self)


class LatLonCS(HorizontalCS):
    """Holds latitude/longitude grid information for both regular and rotated coordinates."""

    def __init__(self, datum, prime_meridian, n_pole, reference_longitude):
        """
        Args:

        * datum:
            An instance of :class:`iris.coord_systems.SpheroidDatum`.
        * prime_meridian:
            An instance of :class:`iris.coord_systems.PrimeMeridian`.
        * n_pole:
            An instance of :class:`iris.coord_systems.GeoPosition` containing the geographic
            location of the, possibly rotated, North pole.
        * reference_longitude:
            The longitude of the standard North pole within the possibly rotated
            coordinate system.
            
        Example creation::
        
            cs = LatLonCS(datum=SpheroidDatum(), 
                          prime_meridian=PrimeMeridian(label="Greenwich", value=0.0), 
                          n_pole=GeoPosition(90, 0), 
                          reference_longitude=0.0
                         )
        
        """
        if n_pole is not None and not isinstance(n_pole, GeoPosition):
            raise TypeError("n_pole must be an instance of GeoPosition")

        HorizontalCS.__init__(self, datum)

        self.datum = datum
        self.prime_meridian = prime_meridian
        self.n_pole = n_pole
        self.reference_longitude = reference_longitude
        self.cs_type = SPHERICAL_CS

    def __repr__(self):
        return "LatLonCS(%r, %r, %r, %r)" % (self.datum, self.prime_meridian, self.n_pole, self.reference_longitude)

    def has_rotated_pole(self):
        return self.n_pole != GeoPosition(90, 0)


class HybridHeightCS(CoordSystem):
    """CoordSystem for holding hybrid height information."""

    def __init__(self, orography):
        """ """
        CoordSystem.__init__(self)
        self.orography = orography

    def __repr__(self):
        return "HybridHeightCS(%r)" % self.orography
    
    def __deepcopy__(self, memo):
        """DON'T duplicate the orography amongst instances - share it."""
        return HybridHeightCS(self.orography)
        
    def orography_at_points(self, cube):
        """ Return a 2D array (YxX) of orography heights for the given cube."""
        if self.orography is None:
            raise TypeError("Regridding cannot be performed as the Orography does not exist (is None).")
        return self.orography.regridded(cube, mode='nearest').data

    def orography_at_xy_corners(self, cube):
        """Return (n, m, 4) array of orography at the 4 lon/lat corners of each cell."""
        # NB. If there are multiple definitive coordinates for an axis it doesn't matter which we use.
        x_coord = cube.coord(axis='x', definitive=True)
        y_coord = cube.coord(axis='y', definitive=True)
        if (not x_coord.has_bounds()) or (not y_coord.has_bounds()):
            raise iris.exceptions.IrisError("x or y coord without bounds")
        if x_coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(x_coord)
        if y_coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(y_coord)
        orography = numpy.empty((y_coord.shape[0], x_coord.shape[0], 4))  # y, x, xyb
        for iy, y_bound in enumerate(y_coord.bounds):
            for ix, x_bound in enumerate(x_coord.bounds):
                # Get the orography at the ll corners for this cell
                interp_value = iris.analysis.interpolate.nearest_neighbour_data_value
                orography[iy, ix, 0] = interp_value(self.orography, {x_coord.name: x_bound[0], y_coord.name: y_bound[0]})
                orography[iy, ix, 1] = interp_value(self.orography, {x_coord.name: x_bound[0], y_coord.name: y_bound[1]})
                orography[iy, ix, 2] = interp_value(self.orography, {x_coord.name: x_bound[1], y_coord.name: y_bound[1]})
                orography[iy, ix, 3] = interp_value(self.orography, {x_coord.name: x_bound[1], y_coord.name: y_bound[0]})
        return orography

    def orography_at_contiguous_corners(self, cube, use_x_bounds, use_y_bounds):
        """Return an (n+1, m), (n, m+1), or (n+1, m+1) array of orography for cell corners/bounds."""
        x_coord = cube.coord(axis='x', definitive=True)
        y_coord = cube.coord(axis='y', definitive=True)
        
        if use_x_bounds:
            x_values = x_coord.contiguous_bounds()
        else:
            x_values = x_coord.points
        if use_y_bounds:
            y_values = y_coord.contiguous_bounds()
        else:
            y_values = y_coord.points

        interp_value = iris.analysis.interpolate.nearest_neighbour_data_value

        orography = numpy.empty((len(y_values), len(x_values)), dtype=self.orography.data.dtype)
        for iy, y in enumerate(y_values):
            for ix, x in enumerate(x_values):
                orography[iy, ix] = interp_value(self.orography, [(x_coord, x), (y_coord, y)])
        return orography

    def _height_3d(self, orography, level_height, sigma):
        """Given a (Y, X) array of orography, return (Z, Y, X) array of heights."""
        # Re-shape the level_height and sigma values so we can use NumPy broadcasting
        # to get our answer in one easy step.
        level_height = numpy.reshape(level_height, (-1, 1, 1))  # z, -, -
        sigma = numpy.reshape(sigma, (-1, 1, 1))  # z, -, -
        return level_height + sigma * orography

    def heights(self, level_height, sigma, cube=None, _orography=None):
        """
        Returns a 3-D array (ZxYxX) of heights above the geoid for the given cube points.

        cube       - defines the points at which we want heights
        _orography - internal optimisation, array of precalculated orography at cube points

        """

        #check params
        if (cube is None) and (_orography is None):
            raise ValueError("No cube specified")
        if (cube is not None) and (_orography is not None):
            raise ValueError("Cannot accept cube and _orography together")

        # Get the orography height for the cell points
        if _orography is None:
            #regrid the orography to the cube's ll grid
            orography = self.orography_at_points(cube)  # y, x
        else:
            #it has already been calculated
            if _orography.ndim != 2:
                raise ValueError("_orography must be 2D")
            orography = _orography

        return self._height_3d(orography, level_height.points, sigma.points)

    def heights_at_contiguous_corners(self, level_height, sigma, cube, use_x_bounds, use_y_bounds):
        """
        Returns a 3-D array (ZxYxX) of heights above the geoid for the given cube points.

        cube         - defines the points at which we want heights
        use_x_bounds - whether we should use point or bound positions along the x axis
        use_y_bounds - whether we should use point or bound positions along the y axis

        """
        orography = self.orography_at_contiguous_corners(cube, use_x_bounds, use_y_bounds) # y, x
        return self._height_3d(orography, level_height.contiguous_bounds(), sigma.contiguous_bounds())
