# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Definitions of coordinate systems.

"""

from abc import ABCMeta  # , abstractmethod
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
    _crs = None

    def __repr__(self):
        return repr(self._crs)

    def __str__(self):
        return str(self._crs)

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

    @classmethod
    def from_crs(cls, crs):
        obj = cls.__new__(cls)
        super(cls, obj).__init__()
        obj._crs = crs
        return obj

    @staticmethod
    def _ellipsoid_to_globe(ellipsoid, globe_default):
        if ellipsoid is not None:
            globe = ellipsoid.as_cartopy_globe()
        else:
            globe = globe_default

        return globe

    def as_cartopy_crs(self):
        """
        Return a cartopy CRS representing our native coordinate
        system.

        """
        return self._crs

    def as_cartopy_projection(self):
        """
        Return a cartopy projection representing our native map.

        This will be the same as the :func:`~CoordSystem.as_cartopy_crs` for
        map projections but for spherical coord systems (which are not map
        projections) we use a map projection, such as PlateCarree.

        """
        if isinstance(self._crs, ccrs.Projection):
            return self._crs
        else:
            print(self._crs)
            print(type(self))
            raise NotImplementedError


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
        datum=None,
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
        #: Describes 'zero' on the ellipsoid in degrees.
        self.longitude_of_prime_meridian = _arg_default(
            longitude_of_prime_meridian, 0
        )

        globe = ccrs.Globe(
            datum=datum,
            ellipse=None,
            semimajor_axis=semi_major_axis,
            semiminor_axis=semi_minor_axis,
            inverse_flattening=inverse_flattening,
        )
        self._crs = ccrs.Geodetic(globe)

    def _pretty_attrs(self):
        attrs = [("semi_major_axis", self.semi_major_axis)]
        if (
            self.semi_major_axis != self.semi_minor_axis
            and self.semi_minor_axis is not None
        ):
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
            return f"GeogCS({float(self.semi_major_axis):.16})"
        else:
            text_attrs = []
            for k, v in attrs:
                if isinstance(v, float):
                    text_attrs.append(f"{k}={float(v):.16}")
                elif isinstance(v, np.float32):
                    text_attrs.append(f"{k}={float(v):.8}")
                else:
                    text_attrs.append(f"{k}={float(v)}")
            return "GeogCS({})".format(", ".join(text_attrs))

    def xml_element(self, doc):
        # Special output for spheres
        attrs = self._pretty_attrs()
        if len(attrs) == 1 and attrs[0][0] == "semi_major_axis":
            attrs = [("earth_radius", float(self.semi_major_axis))]

        return CoordSystem.xml_element(self, doc, attrs)

    def as_cartopy_globe(self):
        # Explicitly set `ellipse` to None as a workaround for
        # Cartopy setting WGS84 as the default.
        return self._crs.globe

    def __getattr__(self, name):
        if name == "semi_major_axis":
            return self._crs.globe.semimajor_axis
        if name == "semi_minor_axis":
            return self._crs.globe.semiminor_axis
        if name == "inverse_flattening":
            return self._crs.globe.inverse_flattening
        if name == "longitude_of_prime_meridian":
            return self._crs.globe.longitude_of_prime_meridian
        if name == "datum":
            return self._crs.datum
        return getattr(super(), name)

    def as_cartopy_projection(self):
        """
        Return a cartopy projection representing our native map.

        This will be the same as the :func:`~CoordSystem.as_cartopy_crs` for
        map projections but for spherical coord systems (which are not map
        projections) we use a map projection, such as PlateCarree.

        """
        return ccrs.PlateCarree(
            central_longitude=self.longitude_of_prime_meridian,
            globe=self._crs.globe,
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

        self._crs = ccrs.RotatedGeodetic(
            grid_north_pole_longitude,
            grid_north_pole_latitude,
            central_rotated_longitude=north_pole_grid_longitude,
            globe=self._ellipsoid_to_globe(ellipsoid, None),
        )

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

    def __getattr__(self, name):
        if name == "grid_north_pole_latitude":
            return self._crs.to_dict()["o_lat_p"]
        if name == "grid_north_pole_longitude":
            return self._crs.to_dict()["lon_0"] - 180
        if name == "north_pole_grid_longitude":
            return self._crs.to_dict()["o_lon_p"]
        if name == "ellipsoid":
            return self._crs.globe.ellipsoid
        if name == "datum":
            return self._crs.datum
        return getattr(super(), name)

    def as_cartopy_projection(self):
        return ccrs.RotatedPole(
            self.grid_north_pole_longitude,
            self.grid_north_pole_latitude,
            self.north_pole_grid_longitude,
            self._crs.globe,
        )


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

        self._crs = ccrs.TransverseMercator(
            central_longitude=longitude_of_central_meridian,
            central_latitude=latitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor=scale_factor_at_central_meridian,
            globe=self._ellipsoid_to_globe(ellipsoid, None),
        )

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

    def __getattr__(self, name):
        if name == "latitude_of_projection_origin":
            return self._crs.to_dict()["lat_0"]
        if name == "longitude_of_central_meridian":
            return self._crs.to_dict()["lon_0"]
        if name == "false_easting":
            return self._crs.to_cf()["false_easting"]
        if name == "false_northing":
            return self._crs.to_cf()["false_northing"]
        if name == "scale_factor_at_central_meridian":
            return self._crs.to_cf()["scale_factor_at_central_meridian"]
        if name == "ellipsoid":
            return self._crs.ellipsoid
        if name == "datum":
            return self._crs.datum
        return getattr(super(), name)


class OSGB(TransverseMercator):
    """A Specific transverse mercator projection on a specific ellipsoid."""

    def __init__(self):
        self._crs = ccrs.OSGB()


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

        self._crs = ccrs.Stereographic(
            central_latitude=central_lat,
            central_longitude=central_lon,
            false_easting=false_easting,
            false_northing=false_northing,
            true_scale_latitude=true_scale_lat,
            scale_factor=None,
            globe=self._ellipsoid_to_globe(ellipsoid, ccrs.Globe()),
        )

    def __getattr__(self, name):
        if name == "central_lat":
            return self._crs.to_dict()["lat_0"]
        if name == "central_lon":
            return self._crs.to_dict()["lon_0"]
        if name == "false_easting":
            return self._crs.to_cf()["false_easting"]
        if name == "false_northing":
            return self._crs.to_cf()["false_northing"]
        if name == "true_scale_lat":
            return None
        if name == "ellipsoid":
            return self._crs.ellipsoid
        if name == "datum":
            return self._crs.datum
        return getattr(super(), name)


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
        # We're either north or south polar. Set a cutoff accordingly.
        if secant_latitudes is not None:
            lats = secant_latitudes
            max_lat = lats[0]
            if len(lats) == 2:
                max_lat = lats[0] if abs(lats[0]) > abs(lats[1]) else lats[1]
            cutoff = -30 if max_lat > 0 else 30
        else:
            cutoff = None

        globe = self._ellipsoid_to_globe(ellipsoid, ccrs.Globe())

        self._crs = ccrs.LambertConformal(
            central_longitude=central_lon,
            central_latitude=central_lat,
            false_easting=false_easting,
            false_northing=false_northing,
            globe=globe,
            cutoff=cutoff,
            standard_parallels=secant_latitudes,
        )

    def __getattr__(self, name):
        if name == "central_lat":
            return self._crs.to_dict()["lat_0"]
        if name == "central_lon":
            return self._crs.to_dict()["lon_0"]
        if name == "false_easting":
            return self._crs.to_cf()["false_easting"]
        if name == "false_northing":
            return self._crs.to_cf()["false_northing"]
        if name == "true_scale_lat":
            return None
        if name == "ellipsoid":
            return self._crs.ellipsoid
        if name == "datum":
            return self._crs.datum
        return getattr(super(), name)

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


class Mercator(CoordSystem):
    """
    A coordinate system in the Mercator projection.

    """

    grid_mapping_name = "mercator"

    def __init__(
        self,
        longitude_of_projection_origin=0,
        ellipsoid=None,
        standard_parallel=0,
        false_easting=0,
        false_northing=0,
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

        * false_easting:
            X offset from the planar origin in metres. Defaults to 0.0.

        * false_northing:
            Y offset from the planar origin in metres. Defaults to 0.0.

        """

        self._crs = ccrs.Mercator(
            central_longitude=longitude_of_projection_origin,
            globe=self._ellipsoid_to_globe(ellipsoid, ccrs.Globe()),
            latitude_true_scale=standard_parallel,
            false_easting=false_easting,
            false_northing=false_northing,
        )

    def __getattr__(self, name):
        if name == "longitude_of_projection_origin":
            return self._crs.to_dict()["lon_0"]
        if name == "standard_parallel":
            return self._crs.to_cf()["standard_parallel"]
        if name == "false_easting":
            return self._crs.to_cf()["false_easting"]
        if name == "false_northing":
            return self._crs.to_cf()["false_northing"]
        if name == "ellipsoid":
            return self._crs.ellipsoid
        if name == "datum":
            return self._crs.datum
        return getattr(super(), name)


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

    def as_cartopy_crs(self):
        globe = self._ellipsoid_to_globe(self.ellipsoid, ccrs.Globe())

        return ccrs.LambertAzimuthalEqualArea(
            central_longitude=self.longitude_of_projection_origin,
            central_latitude=self.latitude_of_projection_origin,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            globe=globe,
        )


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
