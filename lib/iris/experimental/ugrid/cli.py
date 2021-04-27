# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.


import click
from click_default_group import DefaultGroup
from colorama import Fore, Style


# console script name
NAME = f"\n{Fore.GREEN}{Style.DIM}iris-pyvista{Style.RESET_ALL}:"

# camera view planes
VIEW_PLANES = ("xy", "xz", "yx", "yz", "zx", "zy")


def _cli_loader(filename, name):
    from iris import load
    from ..ugrid import PARSE_UGRID_ON_LOAD

    constraints = name if name else None
    with PARSE_UGRID_ON_LOAD.context():
        cube = load(filename, constraints=constraints)
    return cube


def _cli_slice(cube, cube_slice):
    parts = cube_slice.split(",")
    slicer = []

    for part, extent in zip(parts, cube.shape):
        try:
            part = int(part)
            slicer.append(part)
        except ValueError:
            part = part.strip()

            if part == "...":
                emsg = "Ellipsis literal not supported."
                print(emsg)
                exit(1)

            if part == ":":
                slicer.append(slice(None))
            elif ":" in part:
                defaults = dict(start=0, stop=extent, step=1)
                values = part.split(":")
                for key, value in zip(defaults.keys(), values):
                    if value != "":
                        defaults[key] = int(value)
                if (
                    defaults["step"] < 0
                    and defaults["start"] < defaults["stop"]
                ):
                    tmp = defaults["start"]
                    defaults["start"] = defaults["stop"]
                    defaults["stop"] = tmp
                item = slice(*defaults.values())
                slicer.append(item)
            else:
                emsg = (
                    f"{NAME} Unsupported or unknown cube slice syntax, got "
                    '"{cube_slice}".'
                )
                print(emsg)
                exit(1)

    cube = cube[tuple(slicer)]
    return cube


def _cli_threshold(threshold):
    if threshold.lower() in ["true", "yes", "y"]:
        threshold = True
    elif threshold.lower() in ["false", "no", "n"]:
        threshold = False
    else:
        try:
            threshold = float(threshold)
        except ValueError:
            values = threshold.split(",")
            if len(values) != 2:
                emsg = (
                    f"{NAME} Invalid threshold specified, got "
                    '"{threshold}".'
                )
                print(emsg)
                exit(1)
            try:
                threshold = (float(values[0]), float(values[1]))
            except ValueError:
                emsg = (
                    f"{NAME} Invalid threshold value specified, expected "
                    f'"<min>,<max>", got "{threshold}".'
                )
                print(emsg)
                exit(1)

    return threshold


def _cli_view(view, plotter, projection=None):
    if view is not None and view.lower() not in VIEW_PLANES:
        valid = ",".join(VIEW_PLANES)
        emsg = (
            f"{NAME}: Ignoring invalid view plane, expected either "
            f'({valid}), got "{view}".'
        )
        print(emsg)
        view = None

    if view is None:
        view = "xy" if projection else "yz"

    if view == "xy":
        plotter.view_xy()
    elif view == "xz":
        plotter.view_xz()
    elif view == "yx":
        plotter.view_yx()
    elif view == "yz":
        plotter.view_yz()
    elif view == "zx":
        plotter.view_zx()
    else:
        plotter.view_zy()


@click.group(cls=DefaultGroup, default="plot", default_if_no_args=True)
def main():
    """
    To get help for commands, simply use "iris-pyvista <COMMAND> --help".

    """
    pass


#
# cli: plot
#
@click.option(
    "--background-color",
    metavar="<color>",
    type=str,
    help=(
        "Specify the scene background color. Provide either a color name or a "
        "hexidecimal value."
    ),
)
@click.option(
    "--background-image",
    metavar="<filename>",
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Specify the path to an image file to render in the scene "
        "background."
    ),
)
@click.option(
    "--base",
    is_flag=True,
    help="Specify whether to render a base mesh with no associated cell data.",
)
@click.option(
    "--cube-slice",
    metavar="<slice>",
    type=str,
    help=(
        "Comma separated slice used to reduce the dimensionality of the "
        "unstructured cube to either 1D or 2D."
    ),
)
@click.option(
    "-g",
    "--graticule",
    is_flag=True,
    help="Render a labelled graticule of meridian and parallel lines.",
)
@click.option(
    "-i",
    "--invert",
    is_flag=True,
    help=(
        "Invert the nature of the threshold. If threshold is a single value, "
        "then when invert is 'True' cells are kept when their values are "
        "below the threshold. When invert is not specified, cells are kept "
        "when their value is above the threshold."
    ),
)
@click.option(
    "-n",
    "--name",
    metavar="<name>",
    type=str,
    help="Specify the cube name to be applied as a constraint during loading.",
)
@click.option(
    "--no-data",
    is_flag=True,
    help="Don't render the associated unstructured cube data on the mesh.",
)
@click.option(
    "--off-screen",
    metavar="<filename>",
    type=click.Path(),
    help="Render the scene off-screen and save to the specified filename.",
)
@click.option(
    "-p",
    "--projection",
    metavar="<name>",
    type=str,
    help=(
        "The name of the PROJ4 planar projection used to transform the "
        "unstructured cube mesh into a 2D projection coordinate system. "
        "If unspecified, the unstructured cube mesh is rendered on a "
        "3D sphere."
    ),
)
@click.option(
    "-r",
    "--resolution",
    metavar="<value>",
    type=str,
    help=(
        "The resolution of the Natural Earth coastlines, which may be "
        "either '110m', '50m' or '10m'. If unspecified, no coastlines "
        "are rendered."
    ),
)
@click.option(
    "-s",
    "--save",
    metavar="<filename>",
    type=click.Path(),
    help="Save the rendered mesh as VTK format to the specified filename.",
)
@click.option("--show-edges", is_flag=True, help="Render mesh cell edges.")
@click.option(
    "--texture",
    is_flag=True,
    help="Texture map the mesh with a stock map image.",
)
@click.option(
    "--texture-image",
    metavar="<filename>",
    type=click.Path(),
    help="Filename of the image to texture map the mesh.",
)
@click.option(
    "-t",
    "--threshold",
    metavar="<value>",
    default=False,
    show_default=True,
    help=(
        "Apply a threshold to the mesh data. Single value or 'min,max' to "
        "be used for the data threshold. If a sequence, then length must "
        "be 2. If 'True', the non-NaN data range will be used to remove any "
        "NaN values."
    ),
)
@click.option(
    "-v",
    "--view",
    metavar="<plane>",
    type=str,
    help=(
        "Set the view plane of the camera to either 'xy', 'xz', 'yx', "
        "'yz', 'zx' or 'zy'."
    ),
)
@click.argument("filename", type=click.Path(exists=True, dir_okay=False))
@main.command("plot")
def plot(
    background_color,
    background_image,
    base,
    cube_slice,
    graticule,
    invert,
    name,
    no_data,
    off_screen,
    projection,
    resolution,
    save,
    show_edges,
    texture,
    texture_image,
    threshold,
    view,
    filename,
):
    """
    Load a 1D or 2D unstructured Iris cube and render using PyVista.

    """
    import pyvista as pv
    from .plot import plot as ugrid_plot
    from .plot import rcParams

    cubes = _cli_loader(filename, name)

    if len(cubes) != 1:
        plural = "s" if len(cubes) != 1 else ""
        emsg = (
            f"{NAME} Please constrain the load to yield only a single "
            f"cube to plot, got {len(cubes)} cube{plural}.\n"
        )
        print(emsg)
        print(cubes)
        exit(1)

    (cube,) = cubes

    if cube_slice:
        cube = _cli_slice(cube, cube_slice)

    if threshold:
        threshold = _cli_threshold(threshold)

    if show_edges:
        rcParams["plot"]["show_edges"] = True

    plotter = pv.Plotter(off_screen=bool(off_screen))

    return_mesh = True if save else None

    result = ugrid_plot(
        cube,
        projection=projection,
        resolution=resolution,
        texture=texture,
        image=texture_image,
        location=not no_data,
        threshold=threshold,
        invert=invert,
        base=base,
        graticule=graticule,
        plotter=plotter,
        return_mesh=return_mesh,
    )

    _cli_view(view, plotter, projection)

    if background_color:
        plotter.set_background(background_color)
    if background_image:
        plotter.add_background_image(background_image)

    plotter.show_axes()
    plotter.show(screenshot=off_screen)

    if save:
        _, mesh = result
        mesh.save(save)


#
# cli: show
#
@main.command("show")
@click.option(
    "--background-color",
    metavar="<color>",
    type=str,
    help=(
        "Specify the scene background color. Provide either a color name or a "
        "hexidecimal value."
    ),
)
@click.option(
    "--background-image",
    metavar="<filename>",
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Specify the path to an image file to render in the scene "
        "background."
    ),
)
@click.option(
    "-c",
    "--cmap",
    metavar="<name>",
    type=str,
    help=(
        "Select the name of the cmocean, colorcet, or matplotlib colormap "
        "to use when mapping the mesh scalars."
    ),
)
@click.option(
    "--edge-color",
    metavar="<color>",
    default="black",
    show_default=True,
    help=(
        "Solid color to give the mesh cell edges when show-edges is enabled."
        "Provide either a color name or a hexidecimal value."
    ),
)
@click.option(
    "--off-screen",
    metavar="<filename>",
    type=click.Path(),
    help="Render the scene off-screen and save to the specified filename.",
)
@click.option(
    "-s",
    "--scalars",
    metavar="<name>",
    type=str,
    help=(
        "Name of the mesh scalars array for coloring the mesh. "
        "Defaults to the active scalars arrays."
    ),
)
@click.option("--show-edges", is_flag=True, help="Render mesh cell edges.")
@click.option(
    "-v",
    "--view",
    metavar="<plane>",
    type=str,
    help=(
        "Set the view plane of the camera to either 'xy', 'xz', 'yx', "
        "'yz', 'zx' or 'zy'."
    ),
)
@click.argument(
    "filenames", nargs=-1, type=click.Path(exists=True, dir_okay=False)
)
def show(
    background_color,
    background_image,
    cmap,
    edge_color,
    off_screen,
    scalars,
    show_edges,
    view,
    filenames,
):
    """
    Load and render one or more VTK files using PyVista.

    """
    import pyvista as pv

    plotter = pv.Plotter(off_screen=bool(off_screen))
    kwargs = dict(cmap=cmap, show_edges=show_edges, edge_color=edge_color)

    for fname in filenames:
        mesh = pv.read(fname)
        if scalars and scalars in mesh.array_names:
            plotter.add_mesh(mesh, scalars=scalars, **kwargs)
        else:
            plotter.add_mesh(mesh, **kwargs)

    if filenames:
        if background_color:
            plotter.set_background(background_color)
        if background_image:
            plotter.add_background_image(background_image)
        _cli_view(view, plotter)
        plotter.add_axes()
        plotter.show(screenshot=off_screen)
    else:
        emsg = f"{NAME} Please provide at least one VTK file to render."
        print(emsg)


#
# cli: summary
#
@main.command("summary")
@click.argument("filename", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-n",
    "--name",
    metavar="<name>",
    type=str,
    help="Specify the cube name to be applied as a constraint during loading.",
)
def summary(filename, name):
    """
    Print an Iris cube summary.

    """
    cubes = _cli_loader(filename, name)
    if len(cubes) == 1:
        cubes = cubes[0]
    print(cubes)
