# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Plotting support for unstructured meshes using PyVista.

See https://docs.pyvista.org/index.html

"""

from collections.abc import Iterable
from functools import lru_cache
import warnings

import cartopy.io.shapereader as shp
from cartopy.io.shapereader import Record
import numpy as np
import pyvista as pv
from shapely.geometry.multilinestring import MultiLineString
import vtk

from ...config import get_logger


__all__ = [
    "add_coastlines",
    "get_coastlines",
    "plot",
    "to_vtk_mesh",
    "to_xyz",
]

# default to an s2 unit sphere
RADIUS = 1.0

# vtk cell picker callback hook that requires global context
VTK_PICKER_CALLBACK = dict()

# vtk slider widget callback hook that requires global context
VTK_SLIDER_CALLBACK = dict()

# vtk text actor position lookup that requires global context
# see https://vtk.org/doc/nightly/html/classvtkCornerAnnotation.html
VTK_TEXT_POSITIONS = [
    "lower_left",
    "lower_right",
    "upper_left",
    "upper_right",
    "lower_edge",
    "right_edge",
    "left_edge",
    "upper_edge",
]
VTK_POSITIONS_LUT = dict(
    list(zip(VTK_TEXT_POSITIONS, range(len(VTK_TEXT_POSITIONS))))
)

# plotting defaults
rcParams = {
    "add_coastlines": {
        "color": "black",
    },
    "add_slider_widget": {
        "style": "modern",
        "pointa": (0.73, 0.85),
        "pointb": (0.93, 0.85),
    },
    "add_title": {
        "font_size": 12,
        "shadow": True,
    },
    "cell_picking": {
        "style": "points",
        "render_points_as_spheres": True,
        "point_size": 5,
        "show_message": False,
        "font_size": 10,
    },
    "cell_picking_text": {
        "position": "lower_left",
        "font_size": 10,
        "shadow": True,
    },
    "plot": {
        "cmap": "balance",
        "scalar_bar_args": {
            "n_colors": 15,
            "nan_annotation": True,
            "shadow": True,
        },
        "show_edges": False,
        "edge_color": "black",
        "specular": 0,
    },
}

# configure the logger
logger = get_logger(__name__, fmt="[%(cls)s.%(funcName)s]")


class vtkPolyDataTransformFilter:
    """
    A VTK transformer that can project PyVista objects from a latitude and
    longitude projection to a given PROJ4 projection.

    .. seealso::

        * https://proj.org/operations/projections/
        * https://vtk.org/doc/nightly/html/classvtkGeoProjection.html

    """

    def __init__(self, projection_specifier=None):
        """
        Create the VTK transform projection filter.

        Args:

        * projection_specifier:
            The target projection. This may simply be the name of the
            projection e.g., ``moll``, ``sinu``. Alternatively, a
            PROJ4 string may be provided e.g., ``+proj=moll +lon_0=90``

        .. note:: Full PROJ4 string support still requires to be implemented.

        """
        # Set up source and target projection.
        sourceProjection = vtk.vtkGeoProjection()
        destinationProjection = vtk.vtkGeoProjection()
        projection_specifier = projection_specifier.strip()

        if projection_specifier.startswith("+"):
            destinationProjection.SetPROJ4String(projection_specifier)
        else:
            destinationProjection.SetName(projection_specifier)

        # Set up transform between source and target.
        transformProjection = vtk.vtkGeoTransform()
        transformProjection.SetSourceProjection(sourceProjection)
        transformProjection.SetDestinationProjection(destinationProjection)

        # Set up transform filter.
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transformProjection)

        self.transform_filter = transform_filter

    def transform(self, mesh):
        """
        Transform the :class:`~pyvista.core.pointset.PolyData` to the
        target projection.

        Args:

        * mesh (PolyData):
            The :class:~pyvista.core.pointset.PolyData` mesh to be transformed.

        """
        self.transform_filter.SetInputData(mesh)
        self.transform_filter.Update()
        output = self.transform_filter.GetOutput()

        # Wrap output of transform as a PyVista object.
        return pv.wrap(output)


def add_coastlines(resolution="110m", projection=None, plotter=None, **kwargs):
    """
    Add the specified Natural Earth coastline geometries to a PyVista plotter
    for rendering.

    Kwargs:

    * resolution (None or str):
        The resolution of the Natural Earth coastlines, which may be either
        ``110m``, ``50m`` or ``10m``. If ``None``, no coastlines are rendered.
        The default is ``110m``.

    * projection (None or str):
        The name of the PROJ4 planar projection used to transform the coastlines
        into a 2D projection coordinate system. If ``None``, the coastline
        geometries are rendered in on a 3D sphere. The default is ``None``.

    * plotter (None or Plotter):
        The :class:`~pyvista.plotting.plotting.Plotter` which renders the scene.
        If ``None``, a plotter object will be created. Default is None.

    * kwargs (dict):
        Additional ``kwargs`` to be passed to PyVista when creating a coastline
        :class:`~pyvista.core.pointset.PolyData`.

    Returns:
        The :class:`~pyvista.plotting.plotting.Plotter`.

    """
    # use the appropriate pyvista notebook backend
    notebook = is_notebook()
    pv.rcParams["use_ipyvtk"] = notebook

    if plotter is None:
        plotter = pv.Plotter(notebook=notebook)

    defaults = rcParams.get("add_coastlines", {})
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v

    if resolution is not None:
        geocentric = projection is None
        coastlines = get_coastlines(resolution, geocentric=geocentric)

        if projection is not None:
            vtk_projection = vtkPolyDataTransformFilter(projection)
            coastlines = [
                vtk_projection.transform(coastline) for coastline in coastlines
            ]

        for coastline in coastlines:
            plotter.add_mesh(coastline, pickable=False, **kwargs)

    return plotter


@lru_cache
def get_coastlines(resolution="110m", geocentric=False):
    """
    Download and return the collection of Natural Earth coastline geometries.

    The geometries will be transformed appropriately for use with a 2D planar
    projection or a 3D spherical mesh.

    Kwargs:

    * resolution (None or str):
        The resolution of the Natural Earth coastlines, which may be either
        ``110m``, ``50m`` or ``10m``. If ``None``, no coastlines are rendered.
        The default is ``110m``.

    geocentric (bool):
        Convert the coastline latitude and longitude geometries to geocentric
        XYZ coordinates.

    Returns:
        A :class:~pyvista.core.composite.MultiBlock` containing one or more
        :class:`~pyvista.core.pointset.PolyData` coastline geometries.

    """
    # add a "fudge-factor" to ensure coastlines overlay the mesh
    # i.e., a poor mans zorder.
    radius = RADIUS + RADIUS / 1e4

    # load in the shapefiles
    fname = shp.natural_earth(
        resolution=resolution, category="physical", name="coastline"
    )
    reader = shp.Reader(fname)

    dtype = np.float32
    blocks = pv.MultiBlock()
    geoms = []

    def to_pyvista_blocks(records):
        for record in records:
            if isinstance(record, Record):
                geometry = record.geometry
            else:
                geometry = record

            if isinstance(geometry, MultiLineString):
                geoms.extend(list(geometry.geoms))
            else:
                xy = np.array(geometry.coords[:], dtype=dtype)

                if geocentric:
                    # calculate 3d xyz coordinates
                    xr = np.radians(xy[:, 0]).reshape(-1, 1)
                    yr = np.radians(90 - xy[:, 1]).reshape(-1, 1)

                    x = radius * np.sin(yr) * np.cos(xr)
                    y = radius * np.sin(yr) * np.sin(xr)
                    z = radius * np.cos(yr)
                else:
                    # otherwise, calculate xy0 coordinates
                    x = xy[:, 0].reshape(-1, 1)
                    y = xy[:, 1].reshape(-1, 1)
                    z = np.zeros_like(x)

                xyz = np.hstack((x, y, z))
                poly = pv.lines_from_points(xyz, close=False)
                blocks.append(poly)

    to_pyvista_blocks(reader.records())
    to_pyvista_blocks(geoms)

    return blocks


def is_notebook():
    """
    Determine whether we are executing within an ``IPython`` kernel.

    Returns:
        Boolean.

    """
    result = True
    try:
        from IPython import get_ipython

        ip = get_ipython()
        ip.kernel
    except (AttributeError, ModuleNotFoundError):
        result = False
    return result


def add_longitude_labels(plotter, projection=None, step=None):
    if step is None:
        step = 15

    if projection is None:
        lons = np.arange(0, 360, step, dtype=float)
        lons[lons > 180] -= 360
        lats = np.zeros_like(lons)
        points = pv.PolyData(to_xyz(lats, lons))
    else:
        lons = np.arange(0, 360 + step, step, dtype=float)
        lons[lons > 180] -= 360
        lons[-1] = -180
        lats = np.zeros_like(lons)
        xyz = np.vstack([lons, lats, np.zeros_like(lons)]).T
        points = pv.PolyData(xyz)
        vtk_projection = vtkPolyDataTransformFilter(projection)
        points = vtk_projection.transform(points)

    labels = []
    for lon in lons:
        direction = "°E"
        if lon < 0:
            direction = "°W"
        elif lon == 0:
            direction = "°"
        labels.append(f"{int(abs(lon))}{direction}")

    plotter.add_point_labels(
        points, labels, shape=None, bold=False, shadow=True
    )


def add_latitude_labels(
    plotter, projection=None, lat_step=None, lon_step=None
):
    def labels(lats):
        labels = []
        for lat in lats:
            direction = "°N"
            if lat < 0:
                direction = "°S"
            elif lat == 0:
                ""
            labels.append(f"{int(abs(lat))}{direction}")
        return labels

    if lat_step is None:
        lat_step = 15

    if lon_step is None:
        lon_step = 15

    lats = np.arange(0, 180 + lat_step, lat_step, dtype=float)[1:]
    lats[lats > 90] -= 180
    lats[-1] = -90
    lats = sorted(lats)
    points_labels = []
    labels_with_poles = labels(lats)
    labels_without_poles = labels(lats[1:-1])

    if projection is None:
        lons = np.arange(0, 360, lon_step, dtype=float)
        lons[lons > 180] -= 360

        def append_points_labels(lats, lon, without_poles=False):
            points = pv.PolyData(to_xyz(lats, np.ones_like(lats) * lon))
            labels = (
                labels_without_poles if without_poles else labels_with_poles
            )
            points_labels.append((points, labels))

        for without_poles, lon in enumerate(lons):
            append_points_labels(lats, lon, without_poles=bool(without_poles))
            if not without_poles:
                lats = lats[1:-1]
    else:
        lons = np.arange(0, 360 + lon_step, lon_step, dtype=float)
        lons[lons > 180] -= 360
        lons[-1] = -180
        vtk_projection = vtkPolyDataTransformFilter(projection)

        def append_points_labels(lats, lon, without_poles=False):
            xyz = np.vstack(
                [np.ones_like(lats) * lon, lats, np.zeros_like(lats)]
            ).T
            points = vtk_projection.transform(pv.PolyData(xyz))
            labels = (
                labels_without_poles if without_poles else labels_with_poles
            )
            points_labels.append((points, labels))

        for without_poles, lon in enumerate(lons):
            append_points_labels(lats, lon, without_poles=bool(without_poles))
            if not without_poles:
                lats = lats[1:-1]

    for points, labels in points_labels:
        plotter.add_point_labels(
            points, labels, shape=None, bold=False, shadow=True
        )


def namify(item):
    """
    Convenience function that sanitises the name of the provided ``item`` for
    use as human readable text.

    i.e., replace underscores with spaces, and capitalise.

    Args:

    * name (Coord or Cube):
        The instance that requires its associated name to be processed.

    Returns:
        The sanitised name of the instance.

    """
    name = item.name()
    name = (
        " ".join([part.capitalize() for part in name.split("_")])
        if name
        else "Unknown"
    )
    return name


def plot(
    cube,
    projection=None,
    resolution="110m",
    threshold=False,
    invert=False,
    location=True,
    pickable=True,
    cpos=True,
    label_lats=None,
    label_lons=None,
    plotter=None,
    **kwargs,
):
    """
    Plot the cube unstructured mesh using PyVista.

    The cube may be either 1D or 2D, where one dimension is the unstructured
    mesh. For 2D cubes, the other dimension is typically a structured vertical or
    layer type dimension, and a slider widget is rendered for the cube structured
    dimension to allow visualisation of the associated unstructured slices.

    Args:

    * cube (Cube):
        The :class:`~iris.cube.Cube` to be rendered.

    Kwargs:

    * projection (None or str):
        The name of the PROJ4 planar projection used to transform the unstructured
        cube mesh into a 2D projection coordinate system. If ``None``, the unstructured
        cube mesh is rendered on a 3D sphere. The default is ``None``.

    * resolution (None or str):
        The resolution of the Natural Earth coastlines, which may be either
        ``110m``, ``50m`` or ``10m``. If ``None``, no coastlines are rendered.
        The default is ``110m``.

    * threshold (bool or float or sequence):
        Apply a :class:`~pyvista.core.DataSetFilters.threshold`. Single value or
        (min, max) to be used for the data threshold. If a sequence, then length
        must be 2. If ``True``, the non-NaN data range will be used to remove any
        NaN values. Default is ``False``.

    * invert (bool):
        Invert the nature of the ``threshold``. If ``threshold`` is a single value,
        then when invert is ``True`` cells are kept when their values are below
        parameter ``threshold``. When ``invert`` is ``False`` cells are kept when
        their value is above the ``threshold``. Default is ``False``.

    * location (bool):
        Specify whether to add a cell array to the mesh containing the ``location``
        data associated with the unstructured cube. Default is ``True``.

    * pickable (bool):
        Specify whether to enable mesh cell picking. Default is ``True``.

    * cpos (bool or sequence):
        Specify whether to use the default scene camera position, or
        provide the exact camera position to be applied. Default is ``True``.

    * plotter (None or Plotter):
        The :class:`~pyvista.plotting.plotting.Plotter` which renders the scene.
        If ``None``, a plotter object will be created. Default is None.

    * kwargs (dict):
        Additional ``kwargs`` to be passed to PyVista when creating
        :class:`~pyvista.core.pointset.PolyData`.

    Returns:
        The :class:`~pyvista.plotting.plotting.Plotter`.

    """
    global VTK_PICKER_CALLBACK
    global VTK_SLIDER_CALLBACK

    if cube.mesh is None:
        emsg = "Require a cube with an unstructured mesh."
        raise TypeError(emsg)

    if not location:
        pickable = location

    # use the appropriate pyvista notebook backend
    notebook = is_notebook()
    pv.rcParams["use_ipyvtk"] = notebook

    if plotter is None:
        plotter = pv.Plotter(notebook=notebook)

    defaults = rcParams.get("plot", {})
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v

    mesh = to_vtk_mesh(cube, projection=projection, location=location)

    #
    # threshold the mesh, if appropriate
    #
    if isinstance(threshold, bool) and threshold:
        mesh = mesh.threshold(scalars=cube.location, invert=invert)
    elif not isinstance(threshold, bool):
        mesh = mesh.threshold(threshold, scalars=cube.location, invert=invert)
        if isinstance(threshold, (np.ndarray, Iterable)):
            annotations = {threshold[0]: "Lower", threshold[1]: "Upper"}
            if "annotations" not in kwargs:
                kwargs["annotations"] = annotations

    if location and pickable:
        # add unique cell index values to each cell
        mesh.cell_arrays["cids"] = np.arange(mesh.n_cells, dtype=np.uint32)

    if location:
        plotter.add_mesh(
            mesh, scalars=cube.location, pickable=pickable, **kwargs
        )
    else:
        plotter.add_mesh(mesh, pickable=pickable, **kwargs)

    if location:
        add_coastlines(
            resolution=resolution, projection=projection, plotter=plotter
        )

        #
        # scalar bar title
        if (
            "scalar_bar_args" not in kwargs
            or "title" not in kwargs["scalar_bar_args"]
        ):
            name = namify(cube)
            units = str(cube.units)
            plotter.scalar_bar.SetTitle(f"{name} / {units}")

        #
        # text actor for time
        #
        when = None
        coords = cube.coords("time", dimensions=())
        if coords:
            (coord,) = coords
            if coord.units.is_time_reference():
                when = f"Time@{coord.units.num2date(coord.points[0])}"

        coords = cube.coords("forecast_period", dimensions=())
        if coords:
            (coord,) = coords
            if coord.points[0]:
                when = f"{when} ({coord.points[0]} {coord.units})"

        if when:
            plotter.add_text(
                when,
                shadow=True,
                position="upper_left",
                font_size=8,
                name="when",
            )

        #
        # configure mesh cell picking
        #
        if pickable:
            units = (
                ""
                if cube.units.is_unknown()
                or cube.units.is_no_unit()
                or cube.units.is_dimensionless()
                else cube.units
            )

            def picking_callback(mesh):
                global VTK_PICKER_CALLBACK
                global rcParams

                if mesh is not None:
                    text = ""

                    if hasattr(mesh, "cell_arrays"):
                        # cache the cell IDs of the cells that have been picked
                        VTK_PICKER_CALLBACK["cids"] = np.asarray(
                            mesh.cell_arrays["cids"]
                        )
                        # get the location values
                        values = mesh.cell_arrays[
                            VTK_PICKER_CALLBACK["location"]
                        ]
                    else:
                        # the mesh is the cell values i.e., the slider widget
                        # is the caller
                        values = mesh

                    if values.size == 1:
                        if not np.isnan(values[0]):
                            name = namify(cube)
                            name = "Cell" if name == "Unknown" else name
                            text = f"{name} = {values[0]:.2f}{units}"
                    else:
                        with warnings.catch_warnings():
                            # ignore warnings raise by all nan calculations
                            message = "All-NaN slice|Mean of empty slice"
                            warnings.filterwarnings(
                                "ignore",
                                message=message,
                                category=RuntimeWarning,
                            )
                            min, max, mean = (
                                np.nanmin(values),
                                np.nanmax(values),
                                np.nanmean(values),
                            )
                            nans = np.sum(np.isnan(values))
                            tnans = f", NaNs: {nans}" if nans else ""

                        def textify(arg):
                            return (
                                f"{arg}"
                                if np.isnan(arg)
                                else f"{arg:.2f}{units}"
                            )

                        tmin, tmax, tmean = (
                            textify(min),
                            textify(max),
                            textify(mean),
                        )
                        text = f"Min: {tmin}, Max: {tmax}, Mean: {tmean}, nCells: {values.size}{tnans}"

                    defaults = rcParams.get("cell_picking_text", {})

                    if "actor" not in VTK_PICKER_CALLBACK:
                        VTK_PICKER_CALLBACK["actor"] = plotter.add_text(
                            text,
                            name="cell-picking",
                            **defaults,
                        )
                    else:
                        position = defaults.get("position", "lower_left")
                        index = VTK_PICKER_CALLBACK["lut"][position]
                        VTK_PICKER_CALLBACK["actor"].SetText(index, text)

            VTK_PICKER_CALLBACK["location"] = cube.location
            VTK_PICKER_CALLBACK["lut"] = VTK_POSITIONS_LUT
            VTK_PICKER_CALLBACK["callback"] = picking_callback

            defaults = rcParams.get("cell_picking", {})
            plotter.enable_cell_picking(
                through=False,
                callback=picking_callback,
                **defaults,
            )

        #
        # slider for structured dimension, if appropriate
        #
        if cube.ndim == 2:

            def slider_callback(slider, actor):
                global VTK_SLIDER_CALLBACK
                global VTK_PICKER_CALLBACK

                slider = int(slider)

                # only update if the slider value is different
                if slider != VTK_SLIDER_CALLBACK["value"]:
                    sunits = VTK_SLIDER_CALLBACK["sunits"]
                    scoord = VTK_SLIDER_CALLBACK["scoord"]
                    location = VTK_SLIDER_CALLBACK["location"]
                    smesh = VTK_SLIDER_CALLBACK["mesh"]
                    if sunits:
                        if sunits.is_time_reference():
                            slabel = f"{sunits.num2date(scoord[slider])}"
                        else:
                            slabel = f"{scoord[slider]}{sunits}"
                    else:
                        slabel = f"{scoord[slider]}"

                    actor.GetSliderRepresentation().SetLabelFormat(slabel)
                    smesh.cell_arrays[location] = smesh.cell_arrays[
                        f"{location}_{slider}"
                    ]
                    VTK_SLIDER_CALLBACK["value"] = slider

                    # refresh the picker, if available
                    if "cids" in VTK_PICKER_CALLBACK:
                        # get the caches cell IDs of the picked cells
                        cids = VTK_PICKER_CALLBACK["cids"]
                        # get the associated pick cell values
                        picked = smesh.cell_arrays[location][cids]
                        # deal with the single scalar cell case
                        picked = np.array(picked, ndmin=1)
                        # emulate a pick event to refresh
                        VTK_PICKER_CALLBACK["callback"](picked)

            value = 0
            sdim = 1 - cube.mesh_dim()
            scoord = cube.coords(dimensions=(sdim,), dim_coords=True)

            if scoord:
                scoord = scoord[0]
                stitle = namify(scoord)
                nounits = (
                    scoord.units.is_dimensionless()
                    or scoord.units.is_no_unit()
                    or scoord.units.is_dimensionless()
                )
                sunits = "" if nounits else scoord.units
                scoord = scoord.points
            else:
                sunits = stitle = ""
                scoord = np.arange(cube.shape[sdim])

            VTK_SLIDER_CALLBACK["value"] = value
            VTK_SLIDER_CALLBACK["sunits"] = sunits
            VTK_SLIDER_CALLBACK["scoord"] = scoord
            VTK_SLIDER_CALLBACK["location"] = cube.location
            VTK_SLIDER_CALLBACK["mesh"] = mesh

            if sunits:
                if sunits.is_time_reference():
                    slabel = f"{sunits.num2date(scoord[value])}"
                else:
                    slabel = f"{scoord[value]}{sunits}"
            else:
                slabel = f"{scoord[value]}"
            srange = (0, cube.shape[sdim] - 1)

            defaults = rcParams.get("add_slider_widget", {})
            plotter.add_slider_widget(
                slider_callback,
                srange,
                value=value,
                title=stitle,
                event_type="always",
                pass_widget=True,
                fmt=slabel,
                **defaults,
            )

        plotter.show_axes()

    #
    # position the camera on the scene
    #
    if isinstance(cpos, bool):
        if cpos:
            if projection is not None:
                # 2D planar projection camera position
                cpos = [
                    (93959.85410932079, 0.0, 55805210.47284255),
                    (93959.85410932079, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ]
            else:
                # 3D spherical camera position
                cpos = [
                    (5.398270655691156, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                ]

            plotter.camera_position = cpos
    elif cpos:
        plotter.camera_position = cpos

    if location:
        defaults = rcParams.get("add_title", {})
        plotter.add_title(namify(cube), **defaults)

    if label_lons is not None:
        add_longitude_labels(plotter, projection=projection, step=label_lons)

    if label_lats is not None:
        if isinstance(label_lats, Iterable):
            lat_step, lon_step = label_lats
        else:
            lat_step, lon_step = label_lats, None

        add_latitude_labels(
            plotter,
            projection=projection,
            lat_step=lat_step,
            lon_step=lon_step,
        )

    return plotter


def to_vtk_mesh(cube, projection=None, location=True, cids=False):
    """
    Create the PyVista representation of the unstructured cube mesh.

    Args:

    * cube (Cube):
        The :class:`~iris.cube.Cube` to be transformed into a
        :class:`~pyvista.core.pointset.PolyData`.

    Kwargs:

    * projection (None or str):
        The name of the PROJ4 planar projection used to transform the unstructured
        cube mesh into a 2D projection coordinate system. If ``None``, the
        unstructured cube mesh is rendered in a 3D. The default is ``None``.

    * location (bool):
        Specify whether to add a ``cube.location`` cell array to the mesh
        containing the ``location`` data associated with the unstructured cube.
        Default is ``True``.

    * cids (bool):
        Specify whether to add a ``cids`` cell array to the mesh containing a
        unique cell index value to each cell. Default is ``False``.

    Returns:
        The :class:`~pyvista.core.pointset.PolyData`.

    """
    # TBD: deal with generic location of mesh data i.e., support not only face,
    # but also node and edge.

    if cube.mesh is None:
        emsg = "Require a cube with an unstructured mesh."
        raise TypeError(emsg)

    if cube.ndim not in (1, 2):
        emsg = (
            "Require a 1D or 2D cube with an unstructured mesh, "
            f"got a {cube.ndim}D cube."
        )
        raise ValueError(emsg)

    # replace any masks with NaNs
    data = cube.data
    mask = data.mask
    data = data.data
    if np.any(mask):
        data[mask] = np.nan

    # retrieve the mesh topology and connectivity
    face_node = cube.mesh.face_node_connectivity
    indices = face_node.indices_by_src() - face_node.start_index
    coord_x, coord_y = cube.mesh.node_coords

    # TBD: consider masked coordinate points
    node_x = coord_x.points.data
    node_y = coord_y.points.data

    # determine the unstructured dimension (udim) of the cube.
    udim = cube.mesh_dim()

    # simple approach to [-180..180]
    # TBD: ideally we want meridian mesh ripping and re-joining to deal
    # correctly with the mesh connectivity and similar functionality
    # provisioned by iris.analysis.cartography.wrap_lons.
    # however, we need to be careful of circular imports when the plotting
    # code is migrated to its own repo e.g., iris-pyvista
    node_x[node_x > 180] -= 360

    if projection is None:
        # convert lat/lon to geocentric xyz
        xyz = to_xyz(node_y, node_x, vstack=False)
    else:
        # convert lat/lon to planar xy0
        # TBD: deal with mesh splitting for +lon_0
        # TBD: deal with full PROJ4 string
        slicer = [slice(None)] * cube.ndim
        node_z = np.zeros_like(node_y)
        # remove troublesome cells that span seam
        no_wrap = node_x[indices].ptp(axis=-1) < 180
        indices = indices[no_wrap]
        slicer[udim] = no_wrap
        data = data[tuple(slicer)]
        xyz = [node_x, node_y, node_z]

    vertices = np.vstack(xyz).T

    # create connectivity face serialisation i.e., for each quad-mesh face
    # we have (4, C1, C2, C3, C4), where 4 is the number of connectivity
    # indices, followed by the four indices participating in the face.
    N_faces, N_nodes = indices.shape
    faces = np.hstack(
        [
            np.broadcast_to(np.array([N_nodes], np.int8), (N_faces, 1)),
            indices,
        ]
    )

    # create the mesh
    mesh = pv.PolyData(vertices, faces, n_faces=N_faces)

    # add the cube data payload to the mesh as a named "scalar" array
    # based on the location, if required
    if location:
        if cube.ndim == 1:
            mesh.cell_arrays[cube.location] = data
        else:
            # Determine the structured dimension (sdim) of the cube.
            sdim = 1 - udim

            # add the cache of structured dimension data slices to the mesh
            for dim in range(cube.shape[sdim] - 1, -1, -1):
                slicer = [slice(None)] * cube.ndim
                slicer[sdim] = dim
                mesh.cell_arrays[f"{cube.location}_{dim}"] = data[
                    tuple(slicer)
                ]

            mesh.cell_arrays[cube.location] = data[tuple(slicer)]

    # add cell index (cids) to each cell, if required
    if cids:
        mesh.cell_arrays["cids"] = np.arange(mesh.n_cells, dtype=np.uint32)

    # perform the PROJ4 projection of the mesh, if appropriate
    if projection is not None:
        vtk_projection = vtkPolyDataTransformFilter(projection)
        mesh = vtk_projection.transform(mesh)

    return mesh


def to_xyz(latitudes, longitudes, vstack=True):
    """
    Convert latitudes and longitudes to geocentric XYZ values.

    Args:

    * latitudes (float or sequence)
        The latitude values to be converted.

    * longitudes (float or sequence)
        The longitude values to be converted.

    Kwargs:

    * vstack (bool):
        Specify whether the X, Y and Z values are vertically
        stacked and transposed. Default is ``True``.

    Returns:
        The converted latitudes and longitudes.

    """
    latitudes = np.ravel(latitudes)
    longitudes = np.ravel(longitudes)

    x_rad = np.radians(longitudes)
    y_rad = np.radians(90.0 - latitudes)
    x = RADIUS * np.sin(y_rad) * np.cos(x_rad)
    y = RADIUS * np.sin(y_rad) * np.sin(x_rad)
    z = RADIUS * np.cos(y_rad)
    xyz = [x, y, z]

    if vstack:
        xyz = np.vstack(xyz).T

    return xyz
