# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Definitions of coordinate systems.

"""

from abc import ABCMeta, abstractmethod
import warnings

import cartopy.crs as ccrs
import numpy as np


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
    """
    Abstract base class for coordinate systems.

    """

    grid_mapping_name = None

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.__dict__ == other.__dict__
        )

    def __ne__(self, other):
        # Must supply __ne__, Python does not defer to __eq__ for
        # negative equality.
        return not (self == other)

    def xml_element(self, doc, attrs=None):
        """Default behaviour for coord systems."""
        # attrs - optional list of (k,v) items, used for alternate output

        xml_element_name = type(self).__name__
        # lower case the first char
        first_char = xml_element_name[0]
        xml_element_name = xml_element_name.replace(
            first_char, first_char.lower(), 1
        )

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
        """
        Return a cartopy CRS representing our native coordinate
        system.

        """
        pass

    @abstractmethod
    def as_cartopy_projection(self):
        """
        Return a cartopy projection representing our native map.

        This will be the same as the :func:`~CoordSystem.as_cartopy_crs` for
        map projections but for spherical coord systems (which are not map
        projections) we use a map projection, such as PlateCarree.

        """
        pass


class GeogCS(CoordSystem):
    """
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
        """
        Creates a new GeogCS.

        Kwargs:

        * semi_major_axis, semi_minor_axis:
            Axes of ellipsoid, in metres.  At least one must be given
            (see note below).

        * inverse_flattening:
            Can be omitted if both axes given (see note below).
            Defaults to 0.0 .

        * longitude_of_prime_meridian:
            Specifies the prime meridian on the ellipsoid, in degrees.
            Defaults to 0.0 .

        If just semi_major_axis is set, with no semi_minor_axis or
        inverse_flattening, then a perfect sphere is created from the given
        radius.

        If just two of semi_major_axis, semi_minor_axis, and
        inverse_flattening are given the missing element is calculated from the
        formula:
        :math:`flattening = (major - minor) / major`

        Currently, Iris will not allow over-specification (all three ellipsoid
        parameters).

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

        #: Major radius of the ellipsoid in metres.
        self.semi_major_axis = float(semi_major_axis)

        #: Minor radius of the ellipsoid in metres.
        self.semi_minor_axis = float(semi_minor_axis)

        #: :math:`1/f` where :math:`f = (a-b)/a`.
        self.inverse_flattening = float(inverse_flattening)

        #: Describes 'zero' on the ellipsoid in degrees.
        self.longitude_of_prime_meridian = _arg_default(
            longitude_of_prime_meridian, 0
        )

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
        return attrs

    def __repr__(self):
        attrs = self._pretty_attrs()
        # Special case for 1 pretty attr
        if len(attrs) == 1 and attrs[0][0] == "semi_major_axis":
            return "GeogCS(%r)" % self.semi_major_axis
        else:
            return "GeogCS(%s)" % ", ".join(
                ["%s=%r" % (k, v) for k, v in attrs]
            )

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
        return ccrs.Geodetic(self.as_cartopy_globe())

    def as_cartopy_projection(self):
        return ccrs.PlateCarree(
            central_longitude=self.longitude_of_prime_meridian,
            globe=self.as_cartopy_globe(),
        )

    def as_cartopy_globe(self):
        # Explicitly set `ellipse` to None as a workaround for
        # Cartopy setting WGS84 as the default.
        return ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )


class RotatedGeogCS(CoordSystem):
    """
    A coordinate system with rotated pole, on an optional :class:`GeogCS`.

    """

    grid_mapping_name = "rotated_latitude_longitude"

    def __init__(
        self,
        grid_north_pole_latitude,
        grid_north_pole_longitude,
        north_pole_grid_longitude=None,
        ellipsoid=None,
    ):
        """
        Constructs a coordinate system with rotated pole, on an
        optional :class:`GeogCS`.

        Args:

        * grid_north_pole_latitude:
            The true latitude of the rotated pole in degrees.

        * grid_north_pole_longitude:
            The true longitude of the rotated pole in degrees.

        Kwargs:

        * north_pole_grid_longitude:
            Longitude of true north pole in rotated grid, in degrees.
            Defaults to 0.0 .

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        Examples::

            rotated_cs = RotatedGeogCS(30, 30)
            another_cs = RotatedGeogCS(30, 30,
                                       ellipsoid=GeogCS(6400000, 6300000))

        """
        #: The true latitude of the rotated pole in degrees.
        self.grid_north_pole_latitude = float(grid_north_pole_latitude)

        #: The true longitude of the rotated pole in degrees.
        self.grid_north_pole_longitude = float(grid_north_pole_longitude)

        #: Longitude of true north pole in rotated grid in degrees.
        self.north_pole_grid_longitude = _arg_default(
            north_pole_grid_longitude, 0
        )

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

    def _pretty_attrs(self):
        attrs = [
            ("grid_north_pole_latitude", self.grid_north_pole_latitude),
            ("grid_north_pole_longitude", self.grid_north_pole_longitude),
        ]
        if self.north_pole_grid_longitude != 0.0:
            attrs.append(
                ("north_pole_grid_longitude", self.north_pole_grid_longitude)
            )
        if self.ellipsoid is not None:
            attrs.append(("ellipsoid", self.ellipsoid))
        return attrs

    def __repr__(self):
        attrs = self._pretty_attrs()
        result = "RotatedGeogCS(%s)" % ", ".join(
            ["%s=%r" % (k, v) for k, v in attrs]
        )
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
    """
    A cylindrical map projection, with XY coordinates measured in metres.

    """

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
        """
        Constructs a TransverseMercator object.

        Args:

        * latitude_of_projection_origin:
                True latitude of planar origin in degrees.

        * longitude_of_central_meridian:
                True longitude of planar origin in degrees.

        Kwargs:

        * false_easting:
                X offset from planar origin in metres.
                Defaults to 0.0 .

        * false_northing:
                Y offset from planar origin in metres.
                Defaults to 0.0 .

        * scale_factor_at_central_meridian:
                Reduces the cylinder to slice through the ellipsoid
                (secant form). Used to provide TWO longitudes of zero
                distortion in the area of interest.
                Defaults to 1.0 .

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        Example::

            airy1830 = GeogCS(6377563.396, 6356256.909)
            osgb = TransverseMercator(49, -2, 400000, -100000, 0.9996012717,
                                      ellipsoid=airy1830)

        """
        #: True latitude of planar origin in degrees.
        self.latitude_of_projection_origin = float(
            latitude_of_projection_origin
        )

        #: True longitude of planar origin in degrees.
        self.longitude_of_central_meridian = float(
            longitude_of_central_meridian
        )

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: Scale factor at the centre longitude.
        self.scale_factor_at_central_meridian = _arg_default(
            scale_factor_at_central_meridian, 1.0
        )

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

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
    """
    An orthographic map projection.

    """

    grid_mapping_name = "orthographic"

    def __init__(
        self,
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        false_easting=None,
        false_northing=None,
        ellipsoid=None,
    ):
        """
        Constructs an Orthographic coord system.

        Args:

        * latitude_of_projection_origin:
            True latitude of planar origin in degrees.

        * longitude_of_projection_origin:
            True longitude of planar origin in degrees.

        Kwargs:

        * false_easting:
            X offset from planar origin in metres. Defaults to 0.0 .

        * false_northing:
            Y offset from planar origin in metres. Defaults to 0.0 .

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        """
        #: True latitude of planar origin in degrees.
        self.latitude_of_projection_origin = float(
            latitude_of_projection_origin
        )

        #: True longitude of planar origin in degrees.
        self.longitude_of_projection_origin = float(
            longitude_of_projection_origin
        )

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

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
            "not used by Cartopy."
        )

        return ccrs.Orthographic(
            central_longitude=self.longitude_of_projection_origin,
            central_latitude=self.latitude_of_projection_origin,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class VerticalPerspective(CoordSystem):
    """
    A vertical/near-side perspective satellite image map projection.

    """

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
        """
        Constructs a Vertical Perspective coord system.

        Args:

        * latitude_of_projection_origin:
            True latitude of planar origin in degrees.

        * longitude_of_projection_origin:
            True longitude of planar origin in degrees.

        * perspective_point_height:
            Altitude of satellite in metres above the surface of the
            ellipsoid.

        Kwargs:

        * false_easting:
            X offset from planar origin in metres. Defaults to 0.0 .

        * false_northing:
            Y offset from planar origin in metres. Defaults to 0.0 .

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        """
        #: True latitude of planar origin in degrees.
        self.latitude_of_projection_origin = float(
            latitude_of_projection_origin
        )

        #: True longitude of planar origin in degrees.
        self.longitude_of_projection_origin = float(
            longitude_of_projection_origin
        )

        #: Altitude of satellite in metres.
        self.perspective_point_height = float(perspective_point_height)
        # TODO: test if may be cast to float for proj.4

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

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
    """
    A geostationary satellite image map projection.

    """

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

        """
        Constructs a Geostationary coord system.

        Args:

        * latitude_of_projection_origin:
            True latitude of planar origin in degrees.

        * longitude_of_projection_origin:
            True longitude of planar origin in degrees.

        * perspective_point_height:
            Altitude of satellite in metres above the surface of the ellipsoid.

        * sweep_angle_axis (string):
            The axis along which the satellite instrument sweeps - 'x' or 'y'.

        Kwargs:

        * false_easting:
            X offset from planar origin in metres. Defaults to 0.0 .

        * false_northing:
            Y offset from planar origin in metres. Defaults to 0.0 .

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        """
        #: True latitude of planar origin in degrees.
        self.latitude_of_projection_origin = float(
            latitude_of_projection_origin
        )
        if self.latitude_of_projection_origin != 0.0:
            raise ValueError(
                "Non-zero latitude of projection currently not"
                " supported by Cartopy."
            )

        #: True longitude of planar origin in degrees.
        self.longitude_of_projection_origin = float(
            longitude_of_projection_origin
        )

        #: Altitude of satellite in metres.
        self.perspective_point_height = float(perspective_point_height)
        # TODO: test if may be cast to float for proj.4

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: The sweep angle axis (string 'x' or 'y').
        self.sweep_angle_axis = sweep_angle_axis
        if self.sweep_angle_axis not in ("x", "y"):
            raise ValueError('Invalid sweep_angle_axis - must be "x" or "y"')

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

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
    """
    A stereographic map projection.

    """

    grid_mapping_name = "stereographic"

    def __init__(
        self,
        central_lat,
        central_lon,
        false_easting=None,
        false_northing=None,
        true_scale_lat=None,
        ellipsoid=None,
    ):
        """
        Constructs a Stereographic coord system.

        Args:

        * central_lat:
            The latitude of the pole.

        * central_lon:
            The central longitude, which aligns with the y axis.

        Kwargs:

        * false_easting:
            X offset from planar origin in metres. Defaults to 0.0 .

        * false_northing:
            Y offset from planar origin in metres. Defaults to 0.0 .

        * true_scale_lat:
            Latitude of true scale.

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        """

        #: True latitude of planar origin in degrees.
        self.central_lat = float(central_lat)

        #: True longitude of planar origin in degrees.
        self.central_lon = float(central_lon)

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: Latitude of true scale.
        self.true_scale_lat = _arg_default(
            true_scale_lat, None, cast_as=_float_or_None
        )
        # N.B. the way we use this parameter, we need it to default to None,
        # and *not* to 0.0 .

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

    def __repr__(self):
        return (
            "Stereographic(central_lat={!r}, central_lon={!r}, "
            "false_easting={!r}, false_northing={!r}, "
            "true_scale_lat={!r}, "
            "ellipsoid={!r})".format(
                self.central_lat,
                self.central_lon,
                self.false_easting,
                self.false_northing,
                self.true_scale_lat,
                self.ellipsoid,
            )
        )

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.Stereographic(
            self.central_lat,
            self.central_lon,
            self.false_easting,
            self.false_northing,
            self.true_scale_lat,
            globe=globe,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class LambertConformal(CoordSystem):
    """
    A coordinate system in the Lambert Conformal conic projection.

    """

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
        """
        Constructs a LambertConformal coord system.

        Kwargs:

        * central_lat:
                The latitude of "unitary scale".  Defaults to 39.0 .

        * central_lon:
                The central longitude.  Defaults to -96.0 .

        * false_easting:
                X offset from planar origin in metres.  Defaults to 0.0 .

        * false_northing:
                Y offset from planar origin in metres.  Defaults to 0.0 .

        * secant_latitudes (number or iterable of 1 or 2 numbers):
                Latitudes of secant intersection.  One or two.
                Defaults to (33.0, 45.0).

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        .. note:

            Default arguments are for the familiar USA map:
            central_lon=-96.0, central_lat=39.0,
            false_easting=0.0, false_northing=0.0,
            secant_latitudes=(33, 45)

        """

        #: True latitude of planar origin in degrees.
        self.central_lat = _arg_default(central_lat, 39.0)

        #: True longitude of planar origin in degrees.
        self.central_lon = _arg_default(central_lon, -96.0)

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: The standard parallels of the cone (tuple of 1 or 2 floats).
        self.secant_latitudes = _arg_default(
            secant_latitudes, (33, 45), cast_as=_1or2_parallels
        )

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

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
    """
    A coordinate system in the Mercator projection.

    """

    grid_mapping_name = "mercator"

    def __init__(
        self,
        longitude_of_projection_origin=None,
        ellipsoid=None,
        standard_parallel=None,
    ):
        """
        Constructs a Mercator coord system.

        Kwargs:

        * longitude_of_projection_origin:
            True longitude of planar origin in degrees. Defaults to 0.0 .

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        * standard_parallel:
            The latitude where the scale is 1. Defaults to 0.0 .

        """
        #: True longitude of planar origin in degrees.
        self.longitude_of_projection_origin = _arg_default(
            longitude_of_projection_origin, 0
        )

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

        #: The latitude where the scale is 1.
        self.standard_parallel = _arg_default(standard_parallel, 0)

    def __repr__(self):
        res = (
            "Mercator(longitude_of_projection_origin="
            "{self.longitude_of_projection_origin!r}, "
            "ellipsoid={self.ellipsoid!r}, "
            "standard_parallel={self.standard_parallel!r})"
        )
        return res.format(self=self)

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.Mercator(
            central_longitude=self.longitude_of_projection_origin,
            globe=globe,
            latitude_true_scale=self.standard_parallel,
        )

    def as_cartopy_projection(self):
        return self.as_cartopy_crs()


class LambertAzimuthalEqualArea(CoordSystem):
    """
    A coordinate system in the Lambert Azimuthal Equal Area projection.

    """

    grid_mapping_name = "lambert_azimuthal_equal_area"

    def __init__(
        self,
        latitude_of_projection_origin=None,
        longitude_of_projection_origin=None,
        false_easting=None,
        false_northing=None,
        ellipsoid=None,
    ):
        """
        Constructs a Lambert Azimuthal Equal Area coord system.

        Kwargs:

        * latitude_of_projection_origin:
            True latitude of planar origin in degrees. Defaults to 0.0 .

        * longitude_of_projection_origin:
            True longitude of planar origin in degrees. Defaults to 0.0 .

        * false_easting:
                X offset from planar origin in metres. Defaults to 0.0 .

        * false_northing:
                Y offset from planar origin in metres. Defaults to 0.0 .

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        """
        #: True latitude of planar origin in degrees.
        self.latitude_of_projection_origin = _arg_default(
            latitude_of_projection_origin, 0
        )

        #: True longitude of planar origin in degrees.
        self.longitude_of_projection_origin = _arg_default(
            longitude_of_projection_origin, 0
        )

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

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
    """
    A coordinate system in the Albers Conical Equal Area projection.

    """

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
        """
        Constructs a Albers Conical Equal Area coord system.

        Kwargs:

        * latitude_of_projection_origin:
            True latitude of planar origin in degrees. Defaults to 0.0 .

        * longitude_of_central_meridian:
            True longitude of planar central meridian in degrees.
            Defaults to 0.0 .

        * false_easting:
            X offset from planar origin in metres. Defaults to 0.0 .

        * false_northing:
            Y offset from planar origin in metres. Defaults to 0.0 .

        * standard_parallels (number or iterable of 1 or 2 numbers):
            The one or two latitudes of correct scale.
            Defaults to (20.0, 50.0).

        * ellipsoid (:class:`GeogCS`):
            If given, defines the ellipsoid.

        """
        #: True latitude of planar origin in degrees.
        self.latitude_of_projection_origin = _arg_default(
            latitude_of_projection_origin, 0
        )

        #: True longitude of planar central meridian in degrees.
        self.longitude_of_central_meridian = _arg_default(
            longitude_of_central_meridian, 0
        )

        #: X offset from planar origin in metres.
        self.false_easting = _arg_default(false_easting, 0)

        #: Y offset from planar origin in metres.
        self.false_northing = _arg_default(false_northing, 0)

        #: The one or two latitudes of correct scale (tuple of 1 or 2 floats).
        self.standard_parallels = _arg_default(
            standard_parallels, (20, 50), cast_as=_1or2_parallels
        )

        #: Ellipsoid definition (:class:`GeogCS` or None).
        self.ellipsoid = ellipsoid

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
