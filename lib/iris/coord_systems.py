# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Definitions of coordinate systems."""

from abc import ABCMeta, abstractmethod
from functools import cached_property
import re
from typing import ClassVar
import warnings

import cartopy.crs as ccrs
import numpy as np

from iris._deprecation import warn_deprecated
import iris.warnings


def _arg_default(value, default, cast_as=float):
    """Apply a default value and type for an optional kwarg."""
    if value is None:
        value = default
    value = cast_as(value)
    return value


def _1or2_parallels(arg):
    """Accept 1 or 2 inputs as a tuple of 1 or 2 floats."""
    try:
        values_tuple = tuple(arg)
    except TypeError:
        values_tuple = (arg,)
    values_tuple = tuple([float(x) for x in values_tuple])
    nvals = len(values_tuple)
    if nvals not in (1, 2):
        emsg = "Allows only 1 or 2 parallels or secant latitudes : got {!r}"
        raise ValueError(emsg.format(arg))
    return values_tuple


def _float_or_None(arg):
    """Cast as float, except for allowing None as a distinct valid value."""
    if arg is not None:
        arg = float(arg)
    return arg


class CoordSystem(metaclass=ABCMeta):
    """Abstract base class for coordinate systems."""

    grid_mapping_name: ClassVar[str | None] = None

    def __eq__(self, other):
        """Override equality.

        The `_globe` and `_crs` attributes are not compared because they are
        cached properties and completely derived from other attributes. The
        nature of caching means that they can appear on one object and not on
        another despite the objects being identical, and them being completely
        derived from other attributes means they will only differ if other
        attributes that are being tested for equality differ.
        """
        if self.__class__ != other.__class__:
            return False
        self_keys = set(self.__dict__.keys())
        other_keys = set(other.__dict__.keys())
        check_keys = (self_keys | other_keys) - {"_globe", "_crs"}
        for key in check_keys:
            try:
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            except KeyError:
                return False
        return True

    def __ne__(self, other):
        # Must supply __ne__, Python does not defer to __eq__ for
        # negative equality.
        return not (self == other)

    def xml_element(self, doc, attrs=None):
        """Perform default behaviour for coord systems."""
        # attrs - optional list of (k,v) items, used for alternate output

        xml_element_name = type(self).__name__
        # lower case the first char
        first_char = xml_element_name[0]
        xml_element_name = xml_element_name.replace(first_char, first_char.lower(), 1)

        coord_system_xml_element = doc.createElement(xml_element_name)

        if attrs is None:
            attrs = self.__dict__.items()
        attrs = sorted(attrs, key=lambda attr: attr[0])

        for name, value in attrs:
            if isinstance(value, float):
                value_str = "{:.16}".format(value)
            elif isinstance(value, np.float32):
                value_str = "{:.8}".format(value)
            else:
                value_str = "{}".format(value)
            coord_system_xml_element.setAttribute(name, value_str)

        return coord_system_xml_element

    @staticmethod
    def _ellipsoid_to_globe(ellipsoid, globe_default):
        if ellipsoid is not None:
            globe = ellipsoid.as_cartopy_globe()
        else:
            globe = globe_default

        return globe

    @abstractmethod
    def as_cartopy_crs(self):
        """Return a cartopy CRS representing our native coordinate system."""
        pass

    @abstractmethod
    def as_cartopy_projection(self):
        """Return a cartopy projection representing our native map.

        This will be the same as the :func:`~CoordSystem.as_cartopy_crs` for
        map projections but for spherical coord systems (which are not map
        projections) we use a map projection, such as PlateCarree.

        """
        pass


_short_datum_names = {
    "OSGB 1936": "OSGB36",
    "OSGB_1936": "OSGB36",
    "WGS 84": "WGS84",
}


class GeogCS(CoordSystem):
    """A geographic (ellipsoidal) coordinate system.

    A geographic (ellipsoidal) coordinate system, defined by the shape of
    the Earth and a prime meridian.
    """

    grid_mapping_name = "latitude_longitude"

    def __init__(
        self,
        semi_major_axis=None,
        semi_minor_axis=None,
        inverse_flattening=None,
        longitude_of_prime_meridian=None,
    ):
        """Create a new GeogCS.

        Parameters
        ----------
        semi_major_axis, semi_minor_axis : optional
            Axes of ellipsoid, in metres.  At least one must be given (see note
            below).
        inverse_flattening : optional
            Can be omitted if both axes given (see note below). Default 0.0.
        longitude_of_prime_meridian : optional
            Specifies the prime meridian on the ellipsoid, in degrees. Default 0.0.

        Notes
        -----
        If just semi_major_axis is set, with no semi_minor_axis or
        inverse_flattening, then a perfect sphere is created from the given
        radius.

        If just two of semi_major_axis, semi_minor_axis, and inverse_flattening
        are given the missing element is calculated from the formula:
        :math:`flattening = (major - minor) / major`

        Currently, Iris will not allow over-specification (all three ellipsoid
        parameters).

        After object creation, altering any of these properties will not update
        the others. semi_major_axis and semi_minor_axis are used when creating
        Cartopy objects.

        Examples::

            cs = GeogCS(6371229)
            pp_cs = GeogCS(iris.fileformats.pp.EARTH_RADIUS)
            airy1830 = GeogCS(semi_major_axis=6377563.396,
                              semi_minor_axis=6356256.909)
            airy1830 = GeogCS(semi_major_axis=6377563.396,
                              inverse_flattening=299.3249646)
            custom_cs = GeogCS(6400000, 6300000)

        """
        # No ellipsoid specified? (0 0 0)
        if (
            (semi_major_axis is None)
            and (semi_minor_axis is None)
            and (inverse_flattening is None)
        ):
            raise ValueError("No ellipsoid specified")

        # Ellipsoid over-specified? (1 1 1)
        if (
            (semi_major_axis is not None)
            and (semi_minor_axis is not None)
            and (inverse_flattening is not None)
        ):
            raise ValueError("Ellipsoid is overspecified")

        # Perfect sphere (semi_major_axis only)? (1 0 0)
        elif semi_major_axis is not None and (
            semi_minor_axis is None and not inverse_flattening
        ):
            semi_minor_axis = semi_major_axis
            inverse_flattening = 0.0

        # Calculate semi_major_axis? (0 1 1)
        elif semi_major_axis is None and (
            semi_minor_axis is not None and inverse_flattening is not None
        ):
            semi_major_axis = -semi_minor_axis / (
                (1.0 - inverse_flattening) / inverse_flattening
            )

        # Calculate semi_minor_axis? (1 0 1)
        elif semi_minor_axis is None and (
            semi_major_axis is not None and inverse_flattening is not None
        ):
            semi_minor_axis = semi_major_axis - (
                (1.0 / inverse_flattening) * semi_major_axis
            )

        # Calculate inverse_flattening? (1 1 0)
        elif inverse_flattening is None and (
            semi_major_axis is not None and semi_minor_axis is not None
        ):
            if semi_major_axis == semi_minor_axis:
                inverse_flattening = 0.0
            else:
                inverse_flattening = 1.0 / (
                    (semi_major_axis - semi_minor_axis) / semi_major_axis
                )

        # We didn't get enough to specify an ellipse.
        else:
            raise ValueError("Insufficient ellipsoid specification")

        self._semi_major_axis = float(semi_major_axis)
        """Major radius of the ellipsoid in metres."""

        self._semi_minor_axis = float(semi_minor_axis)
        """Minor radius of the ellipsoid in metres."""

        self._inverse_flattening = float(inverse_flattening)
        """:math:`1/f` where :math:`f = (a-b)/a`."""

        self._datum = None

        self.longitude_of_prime_meridian = _arg_default(longitude_of_prime_meridian, 0)
        """Describes 'zero' on the ellipsoid in degrees."""

    def _pretty_attrs(self):
        attrs = [("semi_major_axis", self.semi_major_axis)]
        if self.semi_major_axis != self.semi_minor_axis:
            attrs.append(("semi_minor_axis", self.semi_minor_axis))
        if self.longitude_of_prime_meridian != 0.0:
            attrs.append(
                (
                    "longitude_of_prime_meridian",
                    self.longitude_of_prime_meridian,
                )
            )
        # An unknown crs datum will be treated as None
        if self.datum is not None and self.datum != "unknown":
            attrs.append(
                (
                    "datum",
                    self.datum,
                )
            )
        return attrs

    def __repr__(self):
        attrs = self._pretty_attrs()
        # Special case for 1 pretty attr
        if len(attrs) == 1 and attrs[0][0] == "semi_major_axis":
            return "GeogCS(%r)" % self.semi_major_axis
        else:
            return "GeogCS(%s)" % ", ".join(["%s=%r" % (k, v) for k, v in attrs])

    def __str__(self):
        attrs = self._pretty_attrs()
        # Special case for 1 pretty attr
        if len(attrs) == 1 and attrs[0][0] == "semi_major_axis":
            return "GeogCS({:.16})".format(self.semi_major_axis)
        else:
            text_attrs = []
            for k, v in attrs:
                if isinstance(v, float):
                    text_attrs.append("{}={:.16}".format(k, v))
                elif isinstance(v, np.float32):
                    text_attrs.append("{}={:.8}".format(k, v))
                else:
                    text_attrs.append("{}={}".format(k, v))
            return "GeogCS({})".format(", ".join(text_attrs))

    def xml_element(self, doc):
        # Special output for spheres
        attrs = self._pretty_attrs()
        if len(attrs) == 1 and attrs[0][0] == "semi_major_axis":
            attrs = [("earth_radius", self.semi_major_axis)]

        return CoordSystem.xml_element(self, doc, attrs)

    def as_cartopy_crs(self):
        return self._crs

    def as_cartopy_projection(self):
        return ccrs.PlateCarree(
            central_longitude=self.longitude_of_prime_meridian,
            globe=self.as_cartopy_globe(),
        )

    def as_cartopy_globe(self):
        return self._globe

    @cached_property
    def _globe(self):
        """A representation of this CRS as a Cartopy Globe.

        Note
        ----
        This property is created when required and then cached for speed. That
        cached value is cleared when an assignment is made to a property of the
        class that invalidates the cache.
        """
        if self._datum is not None:
            short_datum = _short_datum_names.get(self._datum, self._datum)
            # Cartopy doesn't actually enact datums unless they're provided without
            # ellipsoid axes, so only provide the datum
            return ccrs.Globe(short_datum, ellipse=None)
        else:
            return ccrs.Globe(
                ellipse=None,
                semimajor_axis=self._semi_major_axis,
                semiminor_axis=self._semi_minor_axis,
            )

    @cached_property
    def _crs(self):
        """A representation of this CRS as a Cartopy CRS.

        Note
        ----
        This property is created when required and then cached for speed. That
        cached value is cleared when an assignment is made to a property of the
        class that invalidates the cache.

        """
        return ccrs.Geodetic(self._globe)

    def _wipe_cached_properties(self):
        """Wipe the cached properties on the object.

        Wipe the cached properties on the object as part of any update to a
        value that invalidates the cache.

        """
        try:
            delattr(self, "_crs")
        except AttributeError:
            pass
        try:
            delattr(self, "_globe")
        except AttributeError:
            pass

    @property
    def semi_major_axis(self):
        if self._semi_major_axis is not None:
            return self._semi_major_axis
        else:
            return self._crs.ellipsoid.semi_major_metre

    @semi_major_axis.setter
    def semi_major_axis(self, value):
        """Assign semi_major_axis.

        Setting this property to a different value invalidates the current datum
        (if any) because a datum encodes a specific semi-major axis. This also
        invalidates the cached `cartopy.Globe` and `cartopy.CRS`.

        """
        value = float(value)
        if not np.isclose(self.semi_major_axis, value):
            self._datum = None
            self._wipe_cached_properties()
        self._semi_major_axis = value

    @property
    def semi_minor_axis(self):
        if self._semi_minor_axis is not None:
            return self._semi_minor_axis
        else:
            return self._crs.ellipsoid.semi_minor_metre

    @semi_minor_axis.setter
    def semi_minor_axis(self, value):
        """Assign semi_minor_axis.

        Setting this property to a different value invalidates the current datum
        (if any) because a datum encodes a specific semi-minor axis. This also
        invalidates the cached `cartopy.Globe` and `cartopy.CRS`.

        """
        value = float(value)
        if not np.isclose(self.semi_minor_axis, value):
            self._datum = None
            self._wipe_cached_properties()
        self._semi_minor_axis = value

    @property
    def inverse_flattening(self):
        if self._inverse_flattening is not None:
            return self._inverse_flattening
        else:
            self._crs.ellipsoid.inverse_flattening

    @inverse_flattening.setter
    def inverse_flattening(self, value):
        """Assign inverse_flattening.

        Setting this property to a different value does not affect the behaviour
        of this object any further than the value of this property.

        """
        wmsg = (
            "Setting inverse_flattening does not affect other properties of "
            "the GeogCS object. To change other properties set them explicitly"
            " or create a new GeogCS instance."
        )
        warnings.warn(wmsg, category=iris.warnings.IrisUserWarning)
        value = float(value)
        self._inverse_flattening = value

    @property
    def datum(self):
        if self._datum is None:
            return None
        else:
            datum = self._datum
            return datum

    @datum.setter
    def datum(self, value):
        """Assign datum.

        Setting this property to a different value invalidates the current
        values of the ellipsoid measurements because a datum encodes its own
        ellipse. This also invalidates the cached `cartopy.Globe` and
        `cartopy.CRS`.

        """
        if self._datum != value:
            self._semi_major_axis = None
            self._semi_minor_axis = None
            self._inverse_flattening = None
            self._wipe_cached_properties()
        self._datum = value

    @classmethod
    def from_datum(cls, datum, longitude_of_prime_meridian=None):
        crs = super().__new__(cls)

        crs._semi_major_axis = None
        crs._semi_minor_axis = None
        crs._inverse_flattening = None

        crs.longitude_of_prime_meridian = _arg_default(longitude_of_prime_meridian, 0)
        """Describes 'zero' on the ellipsoid in degrees."""

        crs._datum = datum

        return crs


class RotatedGeogCS(CoordSystem):
    """A coordinate system with rotated pole, on an optional :class:`GeogCS`."""

    grid_mapping_name = "rotated_latitude_longitude"

    def __init__(
        self,
        grid_north_pole_latitude,
        grid_north_pole_longitude,
        north_pole_grid_longitude=None,
        ellipsoid=None,
    ):
        """Construct a coordinate system with rotated pole, on an optional :class:`GeogCS`.

        Parameters
        ----------
        grid_north_pole_latitude :
            The true latitude of the rotated pole in degrees.
        grid_north_pole_longitude :
            The true longitude of the rotated pole in degrees.
        north_pole_grid_longitude : optional
            Longitude of true north pole in rotated grid, in degrees.
            Defaults to 0.0.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        Examples
        --------
        ::

            rotated_cs = RotatedGeogCS(30, 30)
            another_cs = RotatedGeogCS(30, 30,
                                       ellipsoid=GeogCS(6400000, 6300000))

        """
        self.grid_north_pole_latitude = float(grid_north_pole_latitude)
        """The true latitude of the rotated pole in degrees."""

        self.grid_north_pole_longitude = float(grid_north_pole_longitude)
        """The true longitude of the rotated pole in degrees."""

        self.north_pole_grid_longitude = _arg_default(north_pole_grid_longitude, 0)
        """Longitude of true north pole in rotated grid in degrees."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def _pretty_attrs(self):
        attrs = [
            ("grid_north_pole_latitude", self.grid_north_pole_latitude),
            ("grid_north_pole_longitude", self.grid_north_pole_longitude),
        ]
        if self.north_pole_grid_longitude != 0.0:
            attrs.append(("north_pole_grid_longitude", self.north_pole_grid_longitude))
        if self.ellipsoid is not None:
            attrs.append(("ellipsoid", self.ellipsoid))
        return attrs

    def __repr__(self):
        attrs = self._pretty_attrs()
        result = "RotatedGeogCS(%s)" % ", ".join(["%s=%r" % (k, v) for k, v in attrs])
        # Extra prettiness
        result = result.replace("grid_north_pole_latitude=", "")
        result = result.replace("grid_north_pole_longitude=", "")
        return result

    def __str__(self):
        attrs = self._pretty_attrs()
        text_attrs = []
        for k, v in attrs:
            if isinstance(v, float):
                text_attrs.append("{}={:.16}".format(k, v))
            elif isinstance(v, np.float32):
                text_attrs.append("{}={:.8}".format(k, v))
            else:
                text_attrs.append("{}={}".format(k, v))
        result = "RotatedGeogCS({})".format(", ".join(text_attrs))
        # Extra prettiness
        result = result.replace("grid_north_pole_latitude=", "")
        result = result.replace("grid_north_pole_longitude=", "")
        return result

    def xml_element(self, doc):
        return CoordSystem.xml_element(self, doc, self._pretty_attrs())

    def _ccrs_kwargs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, None)
        cartopy_kwargs = {
            "central_rotated_longitude": self.north_pole_grid_longitude,
            "pole_longitude": self.grid_north_pole_longitude,
            "pole_latitude": self.grid_north_pole_latitude,
            "globe": globe,
        }

        return cartopy_kwargs

    def as_cartopy_crs(self):
        return ccrs.RotatedGeodetic(**self._ccrs_kwargs())

    def as_cartopy_projection(self):
        return ccrs.RotatedPole(**self._ccrs_kwargs())


class TransverseMercator(CoordSystem):
    """A cylindrical map projection, with XY coordinates measured in metres."""

    grid_mapping_name = "transverse_mercator"

    def __init__(
        self,
        latitude_of_projection_origin,
        longitude_of_central_meridian,
        false_easting=None,
        false_northing=None,
        scale_factor_at_central_meridian=None,
        ellipsoid=None,
    ):
        """Construct a TransverseMercator object.

        Parameters
        ----------
        latitude_of_projection_origin :
            True latitude of planar origin in degrees.
        longitude_of_central_meridian :
            True longitude of planar origin in degrees.
        false_easting : optional
            X offset from planar origin in metres.
            Defaults to 0.0.
        false_northing : optional
            Y offset from planar origin in metres.
            Defaults to 0.0.
        scale_factor_at_central_meridian : optional
            Reduces the cylinder to slice through the ellipsoid
            (secant form). Used to provide TWO longitudes of zero
            distortion in the area of interest.
            Defaults to 1.0 .
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        Examples
        --------
        ::

            airy1830 = GeogCS(6377563.396, 6356256.909)
            osgb = TransverseMercator(49, -2, 400000, -100000, 0.9996012717,
                                      ellipsoid=airy1830)

        """
        self.latitude_of_projection_origin = float(latitude_of_projection_origin)
        """True latitude of planar origin in degrees."""

        self.longitude_of_central_meridian = float(longitude_of_central_meridian)
        """True longitude of planar origin in degrees."""

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.scale_factor_at_central_meridian = _arg_default(
            scale_factor_at_central_meridian, 1.0
        )
        """Scale factor at the centre longitude."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "TransverseMercator(latitude_of_projection_origin={!r}, "
            "longitude_of_central_meridian={!r}, false_easting={!r}, "
            "false_northing={!r}, scale_factor_at_central_meridian={!r}, "
            "ellipsoid={!r})".format(
                self.latitude_of_projection_origin,
                self.longitude_of_central_meridian,
                self.false_easting,
                self.false_northing,
                self.scale_factor_at_central_meridian,
                self.ellipsoid,
            )
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, None)

        return ccrs.TransverseMercator(
            central_longitude=self.longitude_of_central_meridian,
            central_latitude=self.latitude_of_projection_origin,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            scale_factor=self.scale_factor_at_central_meridian,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class OSGB(TransverseMercator):
    """A Specific transverse mercator projection on a specific ellipsoid."""

    def __init__(self):
        TransverseMercator.__init__(
            self,
            49,
            -2,
            400000,
            -100000,
            0.9996012717,
            GeogCS(6377563.396, 6356256.909),
        )

    def as_cartopy_crs(self):
        return ccrs.OSGB()

    def as_cartopy_projection(self):
        return ccrs.OSGB()


class Orthographic(CoordSystem):
    """An orthographic map projection."""

    grid_mapping_name = "orthographic"

    def __init__(
        self,
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        false_easting=None,
        false_northing=None,
        ellipsoid=None,
    ):
        """Construct an Orthographic coord system.

        Parameters
        ----------
        latitude_of_projection_origin :
            True latitude of planar origin in degrees.
        longitude_of_projection_origin :
            True longitude of planar origin in degrees.
        false_easting : optional
            X offset from planar origin in metres. Defaults to 0.0.
        false_northing : optional
            Y offset from planar origin in metres. Defaults to 0.0.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        """
        self.latitude_of_projection_origin = float(latitude_of_projection_origin)
        """True latitude of planar origin in degrees."""

        self.longitude_of_projection_origin = float(longitude_of_projection_origin)
        """True longitude of planar origin in degrees."""

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "Orthographic(latitude_of_projection_origin={!r}, "
            "longitude_of_projection_origin={!r}, "
            "false_easting={!r}, false_northing={!r}, "
            "ellipsoid={!r})".format(
                self.latitude_of_projection_origin,
                self.longitude_of_projection_origin,
                self.false_easting,
                self.false_northing,
                self.ellipsoid,
            )
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        warnings.warn(
            "Discarding false_easting and false_northing that are "
            "not used by Cartopy.",
            category=iris.warnings.IrisDefaultingWarning,
        )

        return ccrs.Orthographic(
            central_longitude=self.longitude_of_projection_origin,
            central_latitude=self.latitude_of_projection_origin,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class VerticalPerspective(CoordSystem):
    """A vertical/near-side perspective satellite image map projection."""

    grid_mapping_name = "vertical_perspective"

    def __init__(
        self,
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        perspective_point_height,
        false_easting=None,
        false_northing=None,
        ellipsoid=None,
    ):
        """Construct a Vertical Perspective coord system.

        Parameters
        ----------
        latitude_of_projection_origin :
            True latitude of planar origin in degrees.
        longitude_of_projection_origin :
            True longitude of planar origin in degrees.
        perspective_point_height :
            Altitude of satellite in metres above the surface of the
            ellipsoid.
        false_easting : optional
            X offset from planar origin in metres. Defaults to 0.0.
        false_northing : optional
            Y offset from planar origin in metres. Defaults to 0.0.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        """
        self.latitude_of_projection_origin = float(latitude_of_projection_origin)
        """True latitude of planar origin in degrees."""

        self.longitude_of_projection_origin = float(longitude_of_projection_origin)
        """True longitude of planar origin in degrees."""

        self.perspective_point_height = float(perspective_point_height)
        """Altitude of satellite in metres."""
        # TODO: test if may be cast to float for proj.4

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "Vertical Perspective(latitude_of_projection_origin={!r}, "
            "longitude_of_projection_origin={!r}, "
            "perspective_point_height={!r}, "
            "false_easting={!r}, false_northing={!r}, "
            "ellipsoid={!r})".format(
                self.latitude_of_projection_origin,
                self.longitude_of_projection_origin,
                self.perspective_point_height,
                self.false_easting,
                self.false_northing,
                self.ellipsoid,
            )
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.NearsidePerspective(
            central_latitude=self.latitude_of_projection_origin,
            central_longitude=self.longitude_of_projection_origin,
            satellite_height=self.perspective_point_height,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class Geostationary(CoordSystem):
    """A geostationary satellite image map projection."""

    grid_mapping_name = "geostationary"

    def __init__(
        self,
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        perspective_point_height,
        sweep_angle_axis,
        false_easting=None,
        false_northing=None,
        ellipsoid=None,
    ):
        """Construct a Geostationary coord system.

        Parameters
        ----------
        latitude_of_projection_origin :
            True latitude of planar origin in degrees.
        longitude_of_projection_origin :
            True longitude of planar origin in degrees.
        perspective_point_height :
            Altitude of satellite in metres above the surface of the ellipsoid.
        sweep_angle_axis : str
            The axis along which the satellite instrument sweeps - 'x' or 'y'.
        false_easting : optional
            X offset from planar origin in metres. Defaults to 0.0.
        false_northing : optional
            Y offset from planar origin in metres. Defaults to 0.0.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        """
        self.latitude_of_projection_origin = float(latitude_of_projection_origin)
        """True latitude of planar origin in degrees."""

        if self.latitude_of_projection_origin != 0.0:
            raise ValueError(
                "Non-zero latitude of projection currently not supported by Cartopy."
            )

        self.longitude_of_projection_origin = float(longitude_of_projection_origin)
        """True longitude of planar origin in degrees."""

        self.perspective_point_height = float(perspective_point_height)
        """Altitude of satellite in metres."""
        # TODO: test if may be cast to float for proj.4

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.sweep_angle_axis = sweep_angle_axis
        """The sweep angle axis (string 'x' or 'y')."""

        if self.sweep_angle_axis not in ("x", "y"):
            raise ValueError('Invalid sweep_angle_axis - must be "x" or "y"')

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "Geostationary(latitude_of_projection_origin={!r}, "
            "longitude_of_projection_origin={!r}, "
            "perspective_point_height={!r}, false_easting={!r}, "
            "false_northing={!r}, sweep_angle_axis={!r}, "
            "ellipsoid={!r}".format(
                self.latitude_of_projection_origin,
                self.longitude_of_projection_origin,
                self.perspective_point_height,
                self.false_easting,
                self.false_northing,
                self.sweep_angle_axis,
                self.ellipsoid,
            )
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.Geostationary(
            central_longitude=self.longitude_of_projection_origin,
            satellite_height=self.perspective_point_height,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            globe=globe,
            sweep_axis=self.sweep_angle_axis,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class Stereographic(CoordSystem):
    """A stereographic map projection."""

    grid_mapping_name = "stereographic"

    def __init__(
        self,
        central_lat,
        central_lon,
        false_easting=None,
        false_northing=None,
        true_scale_lat=None,
        ellipsoid=None,
        scale_factor_at_projection_origin=None,
    ):
        """Construct a Stereographic coord system.

        Parameters
        ----------
        central_lat : float
            The latitude of the pole.
        central_lon : float
            The central longitude, which aligns with the y axis.
        false_easting : float, optional
            X offset from planar origin in metres.
        false_northing : float, optional
            Y offset from planar origin in metres.
        true_scale_lat : float, optional
            Latitude of true scale.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.
        scale_factor_at_projection_origin : float, optional
            Scale factor at the origin of the projection.

        Notes
        -----
        It is only valid to provide one of true_scale_lat and
        scale_factor_at_projection_origin

        """
        self.central_lat = float(central_lat)
        """True latitude of planar origin in degrees."""

        self.central_lon = float(central_lon)
        """True longitude of planar origin in degrees."""

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.true_scale_lat = _arg_default(true_scale_lat, None, cast_as=_float_or_None)
        """Latitude of true scale."""

        self.scale_factor_at_projection_origin = _arg_default(
            scale_factor_at_projection_origin, None, cast_as=_float_or_None
        )
        """Scale factor at projection origin."""

        # N.B. the way we use these parameters, we need them to default to None,
        # and *not* to 0.0.

        if (
            self.true_scale_lat is not None
            and self.scale_factor_at_projection_origin is not None
        ):
            raise ValueError(
                "It does not make sense to provide both "
                '"scale_factor_at_projection_origin" and "true_scale_latitude". '
            )

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def _repr_attributes(self):
        if self.scale_factor_at_projection_origin is None:
            scale_info = "true_scale_lat={!r}, ".format(self.true_scale_lat)
        else:
            scale_info = "scale_factor_at_projection_origin={!r}, ".format(
                self.scale_factor_at_projection_origin
            )
        return (
            f"(central_lat={self.central_lat}, central_lon={self.central_lon}, "
            f"false_easting={self.false_easting}, false_northing={self.false_northing}, "
            f"{scale_info}"
            f"ellipsoid={self.ellipsoid})"
        )

    def __repr__(self):
        return "Stereographic" + self._repr_attributes()

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.Stereographic(
            self.central_lat,
            self.central_lon,
            self.false_easting,
            self.false_northing,
            self.true_scale_lat,
            self.scale_factor_at_projection_origin,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class PolarStereographic(Stereographic):
    """A subclass of the stereographic map projection centred on a pole."""

    grid_mapping_name = "polar_stereographic"

    def __init__(
        self,
        central_lat,
        central_lon,
        false_easting=None,
        false_northing=None,
        true_scale_lat=None,
        scale_factor_at_projection_origin=None,
        ellipsoid=None,
    ):
        """Construct a Polar Stereographic coord system.

        Parameters
        ----------
        central_lat : {90, -90}
            The latitude of the pole.
        central_lon : float
            The central longitude, which aligns with the y axis.
        false_easting : float, optional
            X offset from planar origin in metres.
        false_northing : float, optional
            Y offset from planar origin in metres.
        true_scale_lat : float, optional
            Latitude of true scale.
        scale_factor_at_projection_origin : float, optional
            Scale factor at the origin of the projection.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        Notes
        -----
        It is only valid to provide at most one of `true_scale_lat` and
        `scale_factor_at_projection_origin`.


        """
        super().__init__(
            central_lat=central_lat,
            central_lon=central_lon,
            false_easting=false_easting,
            false_northing=false_northing,
            true_scale_lat=true_scale_lat,
            scale_factor_at_projection_origin=scale_factor_at_projection_origin,
            ellipsoid=ellipsoid,
        )

    def __repr__(self):
        return "PolarStereographic" + self._repr_attributes()


class LambertConformal(CoordSystem):
    """A coordinate system in the Lambert Conformal conic projection."""

    grid_mapping_name = "lambert_conformal_conic"

    def __init__(
        self,
        central_lat=None,
        central_lon=None,
        false_easting=None,
        false_northing=None,
        secant_latitudes=None,
        ellipsoid=None,
    ):
        """Construct a LambertConformal coord system.

        Parameters
        ----------
        central_lat : optional
            The latitude of "unitary scale".  Defaults to 39.0 .
        central_lon : optional
            The central longitude.  Defaults to -96.0 .
        false_easting : optional
            X offset from planar origin in metres.  Defaults to 0.0.
        false_northing : optional
            Y offset from planar origin in metres.  Defaults to 0.0.
        secant_latitudes : number or iterable of 1 or 2 numbers, optional
            Latitudes of secant intersection.  One or two.
            Defaults to (33.0, 45.0).
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        Notes
        -----
        .. note:

            Default arguments are for the familiar USA map:
            central_lon=-96.0, central_lat=39.0,
            false_easting=0.0, false_northing=0.0,
            secant_latitudes=(33, 45)

        """
        self.central_lat = _arg_default(central_lat, 39.0)
        """True latitude of planar origin in degrees."""

        self.central_lon = _arg_default(central_lon, -96.0)
        """True longitude of planar origin in degrees."""

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.secant_latitudes = _arg_default(
            secant_latitudes, (33, 45), cast_as=_1or2_parallels
        )
        """tuple: The standard parallels of the cone (tuple of 1 or 2 floats)."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "LambertConformal(central_lat={!r}, central_lon={!r}, "
            "false_easting={!r}, false_northing={!r}, "
            "secant_latitudes={!r}, ellipsoid={!r})".format(
                self.central_lat,
                self.central_lon,
                self.false_easting,
                self.false_northing,
                self.secant_latitudes,
                self.ellipsoid,
            )
        )

    def as_cartopy_crs(self):
        # We're either north or south polar. Set a cutoff accordingly.
        if self.secant_latitudes is not None:
            lats = self.secant_latitudes
            max_lat = lats[0]
            if len(lats) == 2:
                max_lat = lats[0] if abs(lats[0]) > abs(lats[1]) else lats[1]
            cutoff = -30 if max_lat > 0 else 30
        else:
            cutoff = None

        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.LambertConformal(
            central_longitude=self.central_lon,
            central_latitude=self.central_lat,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            globe=globe,
            cutoff=cutoff,
            standard_parallels=self.secant_latitudes,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class Mercator(CoordSystem):
    """A coordinate system in the Mercator projection."""

    grid_mapping_name = "mercator"

    def __init__(
        self,
        longitude_of_projection_origin=None,
        ellipsoid=None,
        standard_parallel=None,
        scale_factor_at_projection_origin=None,
        false_easting=None,
        false_northing=None,
    ):
        """Construct a Mercator coord system.

        Parameters
        ----------
        longitude_of_projection_origin : optional
            True longitude of planar origin in degrees. Defaults to 0.0.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.
        standard_parallel : optional
            The latitude where the scale is 1. Defaults to 0.0.
        scale_factor_at_projection_origin : optional
            Scale factor at natural origin. Defaults to unused.
        false_easting : optional
            X offset from the planar origin in metres. Defaults to 0.0.
        false_northing : optional
            Y offset from the planar origin in metres. Defaults to 0.0.
        datum : optional
            If given, specifies the datumof the coordinate system. Only
            respected if iris.Future.daum_support is set.

        Notes
        -----
        Only one of ``standard_parallel`` and
        ``scale_factor_at_projection_origin`` should be included.

        """
        self.longitude_of_projection_origin = _arg_default(
            longitude_of_projection_origin, 0
        )
        """True longitude of planar origin in degrees."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

        # Initialise to None, then set based on arguments

        self.standard_parallel = None
        """The latitude where the scale is 1."""

        # The scale factor at the origin of the projection
        self.scale_factor_at_projection_origin = None
        if scale_factor_at_projection_origin is None:
            self.standard_parallel = _arg_default(standard_parallel, 0)
        else:
            if standard_parallel is None:
                self.scale_factor_at_projection_origin = _arg_default(
                    scale_factor_at_projection_origin, 0
                )
            else:
                raise ValueError(
                    "It does not make sense to provide both "
                    '"scale_factor_at_projection_origin" and '
                    '"standard_parallel".'
                )

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from the planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from the planar origin in metres."""

    def __repr__(self):
        res = (
            "Mercator(longitude_of_projection_origin="
            "{self.longitude_of_projection_origin!r}, "
            "ellipsoid={self.ellipsoid!r}, "
            "standard_parallel={self.standard_parallel!r}, "
            "scale_factor_at_projection_origin="
            "{self.scale_factor_at_projection_origin!r}, "
            "false_easting={self.false_easting!r}, "
            "false_northing={self.false_northing!r})"
        )
        return res.format(self=self)

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.Mercator(
            central_longitude=self.longitude_of_projection_origin,
            globe=globe,
            latitude_true_scale=self.standard_parallel,
            scale_factor=self.scale_factor_at_projection_origin,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class LambertAzimuthalEqualArea(CoordSystem):
    """A coordinate system in the Lambert Azimuthal Equal Area projection."""

    grid_mapping_name = "lambert_azimuthal_equal_area"

    def __init__(
        self,
        latitude_of_projection_origin=None,
        longitude_of_projection_origin=None,
        false_easting=None,
        false_northing=None,
        ellipsoid=None,
    ):
        """Construct a Lambert Azimuthal Equal Area coord system.

        Parameters
        ----------
        latitude_of_projection_origin : optional
            True latitude of planar origin in degrees. Defaults to 0.0.
        longitude_of_projection_origin : optional
            True longitude of planar origin in degrees. Defaults to 0.0.
        false_easting : optional
            X offset from planar origin in metres. Defaults to 0.0.
        false_northing : optional
            Y offset from planar origin in metres. Defaults to 0.0.
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        """
        self.latitude_of_projection_origin = _arg_default(
            latitude_of_projection_origin, 0
        )
        """True latitude of planar origin in degrees."""

        self.longitude_of_projection_origin = _arg_default(
            longitude_of_projection_origin, 0
        )
        """True longitude of planar origin in degrees."""

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "LambertAzimuthalEqualArea(latitude_of_projection_origin={!r},"
            " longitude_of_projection_origin={!r}, false_easting={!r},"
            " false_northing={!r}, ellipsoid={!r})"
        ).format(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            self.ellipsoid,
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.LambertAzimuthalEqualArea(
            central_longitude=self.longitude_of_projection_origin,
            central_latitude=self.latitude_of_projection_origin,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class AlbersEqualArea(CoordSystem):
    """A coordinate system in the Albers Conical Equal Area projection."""

    grid_mapping_name = "albers_conical_equal_area"

    def __init__(
        self,
        latitude_of_projection_origin=None,
        longitude_of_central_meridian=None,
        false_easting=None,
        false_northing=None,
        standard_parallels=None,
        ellipsoid=None,
    ):
        """Construct a Albers Conical Equal Area coord system.

        Parameters
        ----------
        latitude_of_projection_origin : optional
            True latitude of planar origin in degrees. Defaults to 0.0.
        longitude_of_central_meridian : optional
            True longitude of planar central meridian in degrees.
            Defaults to 0.0.
        false_easting : optional
            X offset from planar origin in metres. Defaults to 0.0.
        false_northing : optional
            Y offset from planar origin in metres. Defaults to 0.0.
        standard_parallels : number or iterable of 1 or 2 numbers, optional
            The one or two latitudes of correct scale.
            Defaults to (20.0, 50.0).
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        """
        self.latitude_of_projection_origin = _arg_default(
            latitude_of_projection_origin, 0
        )
        """True latitude of planar origin in degrees."""

        self.longitude_of_central_meridian = _arg_default(
            longitude_of_central_meridian, 0
        )
        """True longitude of planar central meridian in degrees."""

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.standard_parallels = _arg_default(
            standard_parallels, (20, 50), cast_as=_1or2_parallels
        )
        """The one or two latitudes of correct scale (tuple of 1 or 2 floats)."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "AlbersEqualArea(latitude_of_projection_origin={!r},"
            " longitude_of_central_meridian={!r}, false_easting={!r},"
            " false_northing={!r}, standard_parallels={!r},"
            " ellipsoid={!r})"
        ).format(
            self.latitude_of_projection_origin,
            self.longitude_of_central_meridian,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            self.ellipsoid,
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.AlbersEqualArea(
            central_longitude=self.longitude_of_central_meridian,
            central_latitude=self.latitude_of_projection_origin,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            standard_parallels=self.standard_parallels,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class ObliqueMercator(CoordSystem):
    """A cylindrical map projection, with XY coordinates measured in metres.

    Designed for regions not well suited to :class:`Mercator` or
    :class:`TransverseMercator`, as the positioning of the cylinder is more
    customisable.

    See Also
    --------
    RotatedMercator :
        :class:`ObliqueMercator` with ``azimuth_of_central_line=90``.

    """

    grid_mapping_name = "oblique_mercator"

    def __init__(
        self,
        azimuth_of_central_line,
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        false_easting=None,
        false_northing=None,
        scale_factor_at_projection_origin=None,
        ellipsoid=None,
    ):
        """Construct an ObliqueMercator object.

        Parameters
        ----------
        azimuth_of_central_line : float
            Azimuth of centerline clockwise from north at the center point of
            the centre line.
        latitude_of_projection_origin : float
            The true longitude of the central meridian in degrees.
        longitude_of_projection_origin : float
            The true latitude of the planar origin in degrees.
        false_easting : float, optional
            X offset from the planar origin in metres.
            Defaults to 0.0.
        false_northing : float, optional
            Y offset from the planar origin in metres.
            Defaults to 0.0.
        scale_factor_at_projection_origin : float, optional
            Scale factor at the central meridian.
            Defaults to 1.0 .
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        Examples
        --------
        >>> from iris.coord_systems import GeogCS, ObliqueMercator
        >>> my_ellipsoid = GeogCS(6371229.0, None, 0.0)
        >>> ObliqueMercator(90.0, -22.0, -59.0, -25000.0, -25000.0, 1., my_ellipsoid)
        ObliqueMercator(azimuth_of_central_line=90.0, latitude_of_projection_origin=-22.0, longitude_of_projection_origin=-59.0, false_easting=-25000.0, false_northing=-25000.0, scale_factor_at_projection_origin=1.0, ellipsoid=GeogCS(6371229.0))

        """
        self.azimuth_of_central_line = float(azimuth_of_central_line)
        """Azimuth of centerline clockwise from north."""

        self.latitude_of_projection_origin = float(latitude_of_projection_origin)
        """True latitude of planar origin in degrees."""

        self.longitude_of_projection_origin = float(longitude_of_projection_origin)
        """True longitude of planar origin in degrees."""

        self.false_easting = _arg_default(false_easting, 0)
        """X offset from planar origin in metres."""

        self.false_northing = _arg_default(false_northing, 0)
        """Y offset from planar origin in metres."""

        self.scale_factor_at_projection_origin = _arg_default(
            scale_factor_at_projection_origin, 1.0
        )
        """Scale factor at the central meridian."""

        self.ellipsoid = ellipsoid
        """Ellipsoid definition (:class:`GeogCS` or None)."""

    def __repr__(self):
        return (
            "{!s}(azimuth_of_central_line={!r}, "
            "latitude_of_projection_origin={!r}, "
            "longitude_of_projection_origin={!r}, false_easting={!r}, "
            "false_northing={!r}, scale_factor_at_projection_origin={!r}, "
            "ellipsoid={!r})".format(
                self.__class__.__name__,
                self.azimuth_of_central_line,
                self.latitude_of_projection_origin,
                self.longitude_of_projection_origin,
                self.false_easting,
                self.false_northing,
                self.scale_factor_at_projection_origin,
                self.ellipsoid,
            )
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, None)

        return ccrs.ObliqueMercator(
            central_longitude=self.longitude_of_projection_origin,
            central_latitude=self.latitude_of_projection_origin,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            scale_factor=self.scale_factor_at_projection_origin,
            azimuth=self.azimuth_of_central_line,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class RotatedMercator(ObliqueMercator):
    """:class:`ObliqueMercator` with ``azimuth_of_central_line=90``.

    As noted in CF versions 1.10 and earlier:

        The Rotated Mercator projection is an Oblique Mercator projection
        with azimuth = +90.

    Notes
    -----
    .. deprecated:: 3.8.0
        This coordinate system was introduced as already scheduled for removal
        in a future release, since CF version 1.11 onwards now requires use of
        :class:`ObliqueMercator` with ``azimuth_of_central_line=90.`` .
        Any :class:`RotatedMercator` instances will always be saved to NetCDF
        as the ``oblique_mercator`` grid mapping.

    """

    def __init__(
        self,
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        false_easting=None,
        false_northing=None,
        scale_factor_at_projection_origin=None,
        ellipsoid=None,
    ):
        """Construct a RotatedMercator object.

        Parameters
        ----------
        latitude_of_projection_origin : float
            The true longitude of the central meridian in degrees.
        longitude_of_projection_origin : float
            The true latitude of the planar origin in degrees.
        false_easting : float, optional
            X offset from the planar origin in metres.
            Defaults to 0.0.
        false_northing : float, optional
            Y offset from the planar origin in metres.
            Defaults to 0.0.
        scale_factor_at_projection_origin : float, optional
            Scale factor at the central meridian.
            Defaults to 1.0 .
        ellipsoid : :class:`GeogCS`, optional
            If given, defines the ellipsoid.

        """
        message = (
            "iris.coord_systems.RotatedMercator is deprecated, and will be "
            "removed in a future release. Instead please use "
            "iris.coord_systems.ObliqueMercator with "
            "azimuth_of_central_line=90 ."
        )
        warn_deprecated(message)

        super().__init__(
            90.0,
            latitude_of_projection_origin,
            longitude_of_projection_origin,
            false_easting,
            false_northing,
            scale_factor_at_projection_origin,
            ellipsoid,
        )

    def __repr__(self):
        # Remove the azimuth argument from the parent repr.
        result = super().__repr__()
        result = re.sub(r"azimuth_of_central_line=\d*\.?\d*, ", "", result)
        return result
