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

import iris
import iris.unit


class CoordSystem(object):
    """Abstract base class for coordinate systems.
    
    """
    __metaclass__ = ABCMeta

    name = None
    """CF: grid_mapping_name."""

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)

    def xml_element(self, doc):
        """Default behaviour for coord systems."""
        
        xml_element_name = type(self).__name__
        # lower case the first char
        xml_element_name = xml_element_name.replace(xml_element_name[0], xml_element_name[0].lower(), 1)
        
        coord_system_xml_element = doc.createElement(xml_element_name)
        
        attrs = []
        for k, v in self.__dict__.iteritems():
            attrs.append([k, v])

        attrs.sort(key=lambda attr: attr[0])     

        for name, value in attrs: 
            coord_system_xml_element.setAttribute(name, str(value))
        
        return coord_system_xml_element
    
#     # TODO: Is there value in defining the units this coord system uses? Perhaps for validation purposes.
#     def units(self):
#        raise NotImplementedError() 

    def _to_cartopy(self):
        """Create a representation of ourself using a cartopy object."""
        raise NotImplementedError()


class GeogCS(CoordSystem):
    """An geographic (ellipsoidal) coordinate system, defined by the shape of the Earth and a prime meridian."""
    
    name = "latitude_longitude"
    
    # TODO: Consider including a label, but don't currently see the need.
    def __init__(self, semi_major_axis=None, semi_minor_axis=None, inverse_flattening=None, units=None,
                 longitude_of_prime_meridian=None):
        """Creates a new GeogCS.
        
        Kwargs:
        
            * semi_major_axis              -  of ellipsoid
            * semi_minor_axis              -  of ellipsoid
            * inverse_flattening           -  of ellipsoid 
            * units                        -  of measure of the radii
            * longitude_of_prime_meridian  -  Can be used to specify the prime meridian on the ellipsoid.
                                              Default = 0.  
        
        If all three of semi_major_axis, semi_minor_axis, and inverse_flattening are None then
        it defaults to a perfect sphere using the radius 6371229m.
        
        If just semi_major_axis is set, with no semi_minor_axis or inverse_flattening then
        a perfect sphere is created from the given radius.

        If just two of semi_major_axis, semi_minor_axis, and inverse_flattening are given
        the missing element is calulated from the formula:      
        :math:
        
            flattening = (semi_major_axis - semi_minor_axis) / semi_major_axis
        
        Examples:
        
            default_cs = GeogCS()
            airy1830 = GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909, inverse_flattening=299.3249646, units="m")
            airy1830 = GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909, units="m")
            custom_cs = GeogCS(6400, 6300, units="km")

        """
        # Default (no ellipsoid parmas)
        if (semi_major_axis is None) and (semi_minor_axis is None) and (inverse_flattening is None):
            DEFAULT_RADII = 6371229.0
            semi_major_axis = DEFAULT_RADII
            semi_minor_axis = DEFAULT_RADII
            inverse_flattening = 0.0
            units = iris.unit.Unit('m')

        # Sphere (major axis only)
        elif semi_major_axis is not None and (semi_minor_axis is None and inverse_flattening is None):
            semi_minor_axis = semi_major_axis
            inverse_flattening = 0.0

        # Calculate missing param?
        elif semi_major_axis is None:
            if semi_minor_axis is None or inverse_flattening is None:
                raise ValueError("Must have at least two of semi_major_axis, semi_minor_axis and inverse_flattening")
            semi_major_axis = -semi_minor_axis / ((1.0 - inverse_flattening) / inverse_flattening)

        elif semi_minor_axis is None:
            if semi_major_axis is None or inverse_flattening is None:
                raise ValueError("Must have at least two of semi_major_axis, semi_minor_axis and inverse_flattening")
            semi_minor_axis = semi_major_axis - (1.0 / inverse_flattening) * semi_major_axis

        elif inverse_flattening is None:
            if semi_major_axis is None or semi_minor_axis is None:
                raise ValueError("Must have at least two of semi_major_axis, semi_minor_axis and inverse_flattening")
            if semi_major_axis == semi_minor_axis:
                inverse_flattening = 0.0
            else:
                inverse_flattening = 1.0 / ((semi_major_axis - semi_minor_axis) / semi_major_axis)
            
#        # Validate 3 given ellipsoid params 
#        else:
#            rdiff = (semi_major_axis - semi_minor_axis)
#            if rdiff:
#                result = 1.0 / (rdiff / semi_major_axis)
#                # TODO: assert_almost_equal is not helpful here,
#                # it compares to n decimal places,
#                # not n significant digits.
#                numpy.testing.assert_almost_equal(result, inverse_flattening)
#            elif semi_major_axis == 0.0:
#                raise ValueError("Sphere cannot have zero radius") 
#            elif inverse_flattening != 0.0:
#                raise ValueError("Expected zero inverse_flattening for sphere") 

        self.semi_major_axis = float(semi_major_axis)
        """Major radius of the ellipsoid."""
        
        self.semi_minor_axis = float(semi_minor_axis)
        """Minor radius of the ellipsoid."""

        self.inverse_flattening = float(inverse_flattening)
        """:math:`1/f` where :math:`f = (a-b)/a`"""
        
        self.units = iris.unit.Unit(units)
        """Unit of measure of radii."""
        
        self.longitude_of_prime_meridian = float(longitude_of_prime_meridian) if longitude_of_prime_meridian else 0.0
        """Describes 'zero' on the ellipsoid."""

    def __repr__(self):
        return "GeogCS(semi_major_axis=%r, semi_minor_axis=%r, inverse_flattening=%r, units='%r', longitude_of_prime_meridian=%r)" % \
                    (self.semi_major_axis, self.semi_minor_axis, self.inverse_flattening,
                     self.units, self.longitude_of_prime_meridian)

    def __str__(self):
        return "GeogCS(semi_major_axis=%s, semi_minor_axis=%s, inverse_flattening=%s, units=%s, longitude_of_prime_meridian=%s)" % \
                    (self.semi_major_axis, self.semi_minor_axis, self.inverse_flattening,
                     self.units, self.longitude_of_prime_meridian)

#    def units(self):
#        return "degrees"
        
    def _to_cartopy(self):
        """Create a representation of ourself using a cartopy object."""
        raise NotImplementedError("Cartopy integration not yet implemented")
        


class GeoPos(object):
    """Store the position of the pole, for prettier code."""
    def __init__(self, lat, lon):
        self.lat = float(lat)
        self.lon = float(lon)
    
    def __repr__(self):
        return "GeoPos(%r, %r)" % (self.lat, self.lon)
    
    def __eq__(self, b):
        return self.__class__ == b.__class__ and self.__dict__ == b.__dict__

    def __ne__(self, b):
        return not (self == b)


class RotatedGeogCS(GeogCS):
    """A :class:`GeogCS` with rotated pole."""
    
    name = "rotated_latitude_longitude"
    
    def __init__(self, semi_major_axis=None, semi_minor_axis=None, inverse_flattening=None, units=None,
                 longitude_of_prime_meridian=None, grid_north_pole=None, north_pole_lon=0):
        """For :class:`GeogCS` parameters see :func:`GeogCS.__init__.`

        Args:
        
            * grid_north_pole  -  The true latlon position of the rotated pole: tuple(lat, lon).
                                  CF: (grid_north_pole_latitude, grid_north_ple_longitude).
            
        Kwargs:
        
            * north_pole_lon   -  Longitude of true north pole in rotated grid. Default = 0.
                                  CF: north_pole_grid_longitude

        Example:
        
            rotated_cs = RotatedGeogCS(grid_north_pole=(30,30))
            another_cs = RotatedGeogCS(6400, 6300, units="km", grid_north_pole=(30,30))

        """        
        GeogCS.__init__(self, semi_major_axis, semi_minor_axis, inverse_flattening, units,
                       longitude_of_prime_meridian)

        if grid_north_pole is None:
            raise ValueError("No grid_north_pole specified")
        elif not isinstance(grid_north_pole, GeoPos):
            grid_north_pole = GeoPos(grid_north_pole[0], grid_north_pole[1])


        self.grid_north_pole = grid_north_pole
        """A :class:`~GeoPos` describing the true latlon position of the rotated pole."""
        
        # TODO: Confirm CF's "north_pole_lon" is the same as our old "reference longitude"
        self.north_pole_lon = float(north_pole_lon)
        """Longitude of true north pole in rotated grid."""

    @classmethod
    def from_geocs(cls, geocs, grid_north_pole, north_pole_lon=0):
        """Construct a RotatedGeogCS from a GeogCS. See also :func:`RotatedGeogCS.__init__`"""
        return RotatedGeogCS(geocs.semi_major_axis, geocs.semi_minor_axis, geocs.inverse_flattening, geocs.units,
                            geocs.longitude_of_prime_meridian, grid_north_pole=grid_north_pole, north_pole_lon=north_pole_lon)

    def __repr__(self):
            return "RotatedGeogCS(semi_major_axis=%r, semi_minor_axis=%r, inverse_flattening=%r, units='%r', longitude_of_prime_meridian=%r, grid_north_pole=%r, north_pole_lon=%r)" % \
                        (self.semi_major_axis, self.semi_minor_axis, self.inverse_flattening,
                         self.units, self.longitude_of_prime_meridian,
                         self.grid_north_pole, self.north_pole_lon)

    def __str__(self):
            return "RotatedGeogCS(semi_major_axis=%s, semi_minor_axis=%s, inverse_flattening=%s, units=%s, longitude_of_prime_meridian=%s, grid_north_pole=%s, north_pole_lon=%s)" % \
                        (self.semi_major_axis, self.semi_minor_axis, self.inverse_flattening,
                         self.units, self.longitude_of_prime_meridian,
                         self.grid_north_pole, self.north_pole_lon)

#    def units(self):
#        return "degrees"

    def _to_cartopy(self):
        """Create a representation of ourself using a cartopy object."""
        raise NotImplementedError("Cartopy integration not yet implemented")


class MapProjection(CoordSystem):
    """Abstract base class for map projections.
    
    Describes a transformation from a :class:`GeogCS` to a plane, for mapping.
    A planar coordinate system is produced by the transformation.
    
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, geocs):
        """Creates a MapProjection.
        
        Args:
        
            * geocs  -  A :class:`GeoCs` that describes the Earth, from which we project.
            
        """
        self.geocs = geocs

#     def units(self):
#        raise NotImplementedError() 

    def _to_cartopy(self):
        """Create a representation of ourself using a cartopy object."""
        raise NotImplementedError("Cartopy integration not yet implemented")



class TransverseMercator(MapProjection):
    """A cylindrical map projection, with XY coordinates measured in meters."""

    name = "transverse_mercator" 

    def __init__(self, geocs, origin, false_origin, scale_factor):
        """Constructs a TransverseMercator object.
        
        Args:
        
            * geocs         -  A :class:`GeoCs` that describes the Earth, from which we project.
            * origin        -  True latlon point of planar origin: tuple(lat, lon) in degrees.
                               CF: (latitude_of_projection_origin, longitude_of_central_meridian).
            * false_origin  -  Offset from planar origin: (x, y) in meters.
                               Used to elliminate negative numbers in the area of interest.
                               CF: (false_easting, false_northing).
            * scale_factor  -  Reduces the cylinder to slice through the ellipsoid (secant form).
                               Used to provide TWO longitudes of zero distortion in the area of interest.
                               CF: scale_factor_at_central_meridian.

        Example:
        
            airy1830 = GeogCS(6377563.396, 6356256.910, 299.3249646, "m")
            osgb = TransverseMercator(airy1830, (49,-2), (40000,-10000), 0.9996012717)

        """
        MapProjection.__init__(self, geocs)

        if not isinstance(origin, GeoPos):
            origin = GeoPos(origin[0], origin[1])

        self.geocs = geocs
        
        self.origin = origin
        """True latlon point of planar origin: tuple(lat, lon) in degrees."""
        
        # TODO: Update GeoPos to not just be latlon
        self.false_origin = (float(false_origin[0]), float(false_origin[1]))  
        """Offset from planar origin: (x, y) in meters."""
        
        self.scale_factor = float(scale_factor)
        """Reduces the cylinder to slice through the ellipsoid."""

    def __repr__(self):
        return "TransverseMercator(origin=%r, false_origin=%r, scale_factor=%r, geos=%r)" % \
                    (self.origin, self.false_origin, self.scale_factor, self.geocs)

#    def units(self):
#        return "meters"

    def _to_cartopy(self):
        """Create a representation of ourself using a cartopy object."""
        raise NotImplementedError("Cartopy integration not yet implemented")
