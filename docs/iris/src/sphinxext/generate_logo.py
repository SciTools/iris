# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Script to generate the Iris logo in every required format.
Uses XML ElementTree for SVG file editing.
"""

from copy import deepcopy
from io import BytesIO
from os import environ
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom
from zipfile import ZipFile

from cartopy import crs as ccrs
from cartopy.feature import LAND
from matplotlib import pyplot as plt
import numpy as np

print("LOGO GENERATION START ...")

################################################################################
# Configuration

# How much bigger than the globe the iris clip should be.
CLIP_GLOBE_RATIO = 1.28

# Pixel size of the largest dimension of the main logo.
# (Proportions dictated by those of the clip).
LOGO_PIXELS = 1024

# Banner width, text and text size must be manually tuned to work together.
# Pixel dimensions of text banner.
BANNER_PIXELS = {"width": 588, "height": 256}
# Text printed in the banner.
BANNER_TEXT = "Iris"
# How much smaller than the globe the banner text should be.
TEXT_GLOBE_RATIO = 0.6

WRITE_DIRECTORY = Path("_static")
# Start of all filenames to be written.
FILENAME_PREFIX = environ["PROJECT_PREFIX"]

# The logo's SVG elements can be configured at their point of definition below.

################################################################################
# Create new SVG elements for logo.

# XML ElementTree setup.
namespaces = {"svg": "http://www.w3.org/2000/svg"}
ET.register_namespace("", namespaces["svg"])

# The elements that are displayed (not just referenced).
# Order is important for layering.
artwork_dict = dict.fromkeys(["background", "glow", "sea", "land",])
# The elements that will just be referenced by artwork elements.
defs_dict = {}

# CLIP ####################
# SVG string representing bezier curves, drawn in a GUI then size-converted
# for this file.
clip_size_xy_original = np.array([133.334, 131.521])
clip_scaling = LOGO_PIXELS / max(clip_size_xy_original)

# Use clip proportions to dictate logo proportions and dimensions.
logo_proportions_xy = clip_size_xy_original / max(clip_size_xy_original)
logo_size_xy = logo_proportions_xy * LOGO_PIXELS
logo_centre_xy = logo_size_xy / 2

clip_string = "M 66.582031,73.613283 C 62.196206,73.820182 51.16069," \
              "82.105643 33.433594,80.496096 18.711759,79.15939 10.669958," \
              "73.392913 8.4375,74.619143 7.1228015,75.508541 6.7582985," \
              "76.436912 6.703125,78.451174 6.64868,80.465549 5.5985568," \
              "94.091611 12.535156,107.18359 c 8.745259,16.50554 21.06813," \
              "20.14551 26.152344,20.24414 12.671346,0.24601 24.681745," \
              "-21.45054 27.345703,-30.62304 3.143758,-10.82453 4.457007," \
              "-22.654297 1.335938,-23.119141 -0.233298,-0.06335 -0.494721," \
              "-0.08606 -0.78711,-0.07227 z m 7.119141,-5.857422 c " \
              "-0.859687,0.02602 -1.458448,0.280361 -1.722656,0.806641 " \
              "-2.123362,3.215735 3.515029,16.843803 -3.970704,34.189448 " \
              "-5.828231,13.50478 -13.830895,19.32496 -13.347656,21.81446 " \
              "0.444722,1.5177 1.220596,2.14788 3.13086,2.82226 1.910471," \
              "0.67381 14.623496,5.87722 29.292968,3.36524 18.494166," \
              "-3.16726 25.783696,-13.69084 27.449216,-18.4668 C 118.68435," \
              "100.38386 101.63623,82.323612 93.683594,76.970705 86.058314," \
              "71.837998 77.426483,67.643118 73.701172,67.755861 Z M " \
              "114.02539,33.224611 C 103.06524,33.255401 88.961605,40.28151 " \
              "83.277344,44.67969 74.333356,51.599967 66.27534,60.401955 " \
              "68.525391,62.601564 c 2.420462,3.001097 17.201948,1.879418 " \
              "31.484379,14.316407 11.11971,9.683149 14.21486,19.049139 " \
              "16.74609,19.361328 1.58953,0.0486 2.43239,-0.488445 3.66797," \
              "-2.085938 1.23506,-1.597905 10.14187,-12.009723 12.27148," \
              "-26.654297 2.68478,-18.462878 -5.13068,-28.607634 -9.18554," \
              "-31.658203 -2.52647,-1.900653 -5.831,-2.666512 -9.48438," \
              "-2.65625 z M 39.621094,14.64258 C 39.094212,14.665 38.496575," \
              "14.789793 37.767578,15.003908 35.823484,15.574873 22.460486," \
              "18.793044 12.078125,29.396486 -1.0113962,42.764595 " \
              "-0.68566506,55.540029 0.79101563,60.376955 4.4713185," \
              "72.432363 28.943765,77.081596 38.542969,76.765627 49.870882," \
              "76.392777 61.593892,73.978953 61.074219,70.884768 60.890477," \
              "67.042613 48.270811,59.312854 44.070312,40.906252 40.799857," \
              "26.575361 43.83381,17.190581 41.970703,15.458986 41.184932," \
              "14.853981 40.49923,14.605209 39.621094,14.64258 Z M " \
              "67.228516,0.08984563 C 60.427428,0.11193533 55.565192," \
              "2.1689455 53.21875,3.7949238 42.82192,10.999553 45.934544," \
              "35.571547 49.203125,44.54883 c 3.857276,10.594054 9.790034," \
              "20.931896 12.589844,19.484375 3.619302,-1.360857 7.113072," \
              "-15.680732 23.425781,-25.339844 12.700702,-7.520306 22.61812," \
              "-7.553206 23.69922,-9.849609 0.53758,-1.487663 0.28371," \
              "-2.45185 -0.86328,-4.113281 C 106.90759,23.069051 99.699016," \
              "11.431303 86.345703,4.89258 78.980393,1.2860119 72.51825," \
              "0.07266475 67.228516,0.08984563 Z"

iris_clip = ET.Element("clipPath")
iris_clip.append(ET.Element("path", attrib={"d": clip_string,
                                            "transform": f"scale({clip_scaling})"}))
defs_dict["iris_clip"] = iris_clip

# BACKGROUND
artwork_dict["background"] = ET.Element(
    "rect",
    attrib={
        "height": "100%",
        "width": "100%",
        "fill": "url(#background_gradient)",
    },
)
background_gradient = ET.Element(
    "linearGradient", attrib={"y1": "0%", "y2": "100%",},
)
background_gradient.append(
    ET.Element("stop", attrib={"offset": "0", "stop-color": "#13385d",},)
)
background_gradient.append(
    ET.Element("stop", attrib={"offset": "0.43", "stop-color": "#0b3849",},)
)
background_gradient.append(
    ET.Element("stop", attrib={"offset": "1", "stop-color": "#272b2c",},)
)
defs_dict["background_gradient"] = background_gradient

# LAND ####################
# (Using Matplotlib and Cartopy).

# Set plotting size/proportions.
mpl_points_per_inch = 72
plot_inches = logo_size_xy / mpl_points_per_inch
plot_padding = (1 - (1 / CLIP_GLOBE_RATIO)) / 2

# Create land with simplified coastlines.
simple_geometries = [
    geometry.simplify(0.8, True) for geometry in LAND.geometries()
]
LAND.geometries = lambda: iter(simple_geometries)

# Variable that will store the sequence of land-shaped SVG clips for each longitude.
land_clips = []

# Create a sequence of longitude values.
central_longitude = -30
central_latitude = 22.9
perspective_tilt = -4.99
rotation_frames = 180
rotation_longitudes = np.linspace(start=central_longitude + 360,
                                  stop=central_longitude,
                                  num=rotation_frames,
                                  endpoint=False)
# Normalise to -180..+180
rotation_longitudes = (rotation_longitudes + 360.0 + 180.0) % 360.0 - 180.0

transform_string = f"rotate({perspective_tilt} {logo_centre_xy[0]} {logo_centre_xy[1]})"

for lon in rotation_longitudes:
    # Use Matplotlib and Cartopy to generate land-shaped SVG clips for each longitude.

    projection_rotated = ccrs.Orthographic(central_longitude=lon,
                                           central_latitude=central_latitude)

    # Use constants set earlier to achieve desired dimensions.
    fig = plt.figure(0, figsize=plot_inches)
    ax = plt.subplot(projection=projection_rotated)
    plt.subplots_adjust(left=plot_padding, bottom=plot_padding,
                        right=1 - plot_padding, top=1 - plot_padding)
    ax.add_feature(LAND)

    # Save as SVG and extract the resultant code.
    svg_bytes = BytesIO()
    plt.savefig(svg_bytes, format="svg")
    svg_mpl = ET.fromstring(svg_bytes.getvalue())

    # Find land paths and convert to clip paths.
    mpl_land = svg_mpl.find(".//svg:g[@id='figure_1']", namespaces)
    land_paths = mpl_land.find(".//svg:g[@id='PathCollection_1']", namespaces)
    for path in land_paths:
        # Remove all other attribute items.
        path.attrib = {"d": path.attrib["d"], "stroke-linejoin": "round"}
    land_paths.tag = "clipPath"
    land_paths.attrib["transform"] = transform_string
    land_clips.append(land_paths)

# Extract the final land clip for use as the default.
defs_dict["land_clip"] = land_clips[-1]

artwork_dict["land"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": f"{50 / CLIP_GLOBE_RATIO}%",
        "fill": "url(#land_gradient)",
        "clip-path": "url(#land_clip)",
    },
)
land_gradient = ET.Element("radialGradient")
land_gradient.append(
    ET.Element("stop", attrib={"offset": "0", "stop-color": "#d5e488"},)
)
land_gradient.append(
    ET.Element("stop", attrib={"offset": "1", "stop-color": "#aec928",},)
)
defs_dict["land_gradient"] = land_gradient

# SEA #####################
# Not using Cartopy for sea since it doesn't actually render curves/circles.
artwork_dict["sea"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": f"{50.5 / CLIP_GLOBE_RATIO}%",
        "fill": "url(#sea_gradient)",
    },
)
sea_gradient = ET.Element("radialGradient")
sea_gradient.append(
    ET.Element("stop", attrib={"offset": "0", "stop-color": "#20b0ea"},)
)
sea_gradient.append(
    ET.Element("stop", attrib={"offset": "1", "stop-color": "#156475",},)
)
defs_dict["sea_gradient"] = sea_gradient

# GLOW ####################
artwork_dict["glow"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": f"{52 / CLIP_GLOBE_RATIO}%",
        "fill": "url(#glow_gradient)",
        "filter": "url(#glow_blur)",
        "stroke": "#ffffff",
        "stroke-width": "2",
        "stroke-opacity": "0.797414",
    },
)
glow_gradient = ET.Element(
    "radialGradient",
    attrib={"gradientTransform": "scale(1.15, 1.35), translate(-0.1, -0.3)"},
)
glow_gradient.append(
    ET.Element(
        "stop",
        attrib={
            "offset": "0",
            "stop-color": "#0aaea7",
            "stop-opacity": "0.85882354",
        },
    )
)
glow_gradient.append(
    ET.Element(
        "stop",
        attrib={
            "offset": "0.67322218",
            "stop-color": "#18685d",
            "stop-opacity": "0.74117649",
        },
    )
)
glow_gradient.append(
    ET.Element("stop", attrib={"offset": "1", "stop-color": "#b6df34",},)
)
defs_dict["glow_gradient"] = glow_gradient

glow_blur = ET.Element("filter")
glow_blur.append(ET.Element("feGaussianBlur", attrib={"stdDeviation": "14"}))
defs_dict["glow_blur"] = glow_blur

################################################################################
# Create SVG's


def svg_logo(defs_dict, artwork_dict):
    # Group contents into a logo subgroup (so text can be stored separately).
    logo_group = ET.Element("svg", attrib={"id": "logo_group"})
    logo_group.attrib["viewBox"] = f"0 0 {logo_size_xy[0]} {logo_size_xy[1]}"

    def populate_element_group(group, children_dict):
        """Write each element from a dictionary, assigning an appropriate ID."""
        for name, element in children_dict.items():
            element.attrib["id"] = name
            group.append(element)
        logo_group.append(group)

    defs_element = ET.Element("defs")
    populate_element_group(defs_element, defs_dict)

    # All artwork is clipped by the Iris shape.
    artwork_element = ET.Element("g", attrib={"clip-path": "url(#iris_clip)"})
    populate_element_group(artwork_element, artwork_dict)

    root = ET.Element("svg")
    for ix, dim in enumerate(("width", "height")):
        root.attrib[dim] = str(logo_size_xy[ix])
    root.append(logo_group)

    return root


def svg_banner(logo_svg):
    banner_height = BANNER_PIXELS["height"]
    text_size = banner_height * TEXT_GLOBE_RATIO
    text_x = banner_height + 8
    # Manual y centring since SVG dominant-baseline not widely supported.
    text_y = banner_height - (banner_height - text_size) / 2
    text_y *= 0.975  # Slight offset

    text = ET.Element(
        "text",
        attrib={
            "x": str(text_x),
            "y": str(text_y),
            "font-size": f"{text_size}pt",
            "font-family": "georgia",
        },
    )
    text.text = BANNER_TEXT

    root = deepcopy(logo_svg)
    for dimension, pixels in BANNER_PIXELS.items():
        root.attrib[dimension] = str(pixels)

    # Left-align the logo.
    banner_logo_group = root.find("svg", namespaces)
    banner_logo_group.attrib["preserveAspectRatio"] = "xMinYMin meet"

    root.append(text)

    return root


logo = svg_logo(defs_dict=defs_dict, artwork_dict=artwork_dict)
banner = svg_banner(logo)

################################################################################
# Write files.


def write_svg_file(svg_root, filename_suffix, zip_archive=None):
    """Format the svg then write the svg to a file in WRITE_DIRECTORY, or
    optionally to an open ZipFile."""
    input_string = ET.tostring(svg_root)
    pretty_xml = minidom.parseString(input_string).toprettyxml()
    # Remove extra empty lines from Matplotlib.
    pretty_xml = "\n".join(
        [line for line in pretty_xml.split("\n") if line.strip()]
    )

    filename = f"{FILENAME_PREFIX}-{filename_suffix}.svg"
    if isinstance(zip_archive, ZipFile):
        zip_archive.writestr(filename, pretty_xml)
    else:
        write_path = WRITE_DIRECTORY.joinpath(filename)
        with open(write_path, "w") as f:
            f.write(pretty_xml)


def replace_land_clip(svg_root, new_clip):
    new_root = deepcopy(svg_root)
    new_clip.attrib["id"] = "land_clip"

    defs = new_root.find("svg/defs")
    land_clip = defs.find(".//clipPath[@id='land_clip']")
    defs.remove(land_clip)
    defs.append(new_clip)

    return new_root


write_dict = {
    "logo": logo,
    "logo-title": banner,
}
for suffix, svg in write_dict.items():
    write_svg_file(svg, suffix)

    # Zip archive containing components for manual creation of rotating logo.
    zip_path = WRITE_DIRECTORY.joinpath(f"{suffix}_rotate.zip")
    with ZipFile(zip_path, "w") as rotate_zip:
        for ix, clip in enumerate(land_clips):
            svg_rotated = replace_land_clip(svg, clip)
            write_svg_file(svg_rotated, f"{suffix}_rotate{ix:03d}", rotate_zip)

        readme_str = "Several tools are available to stitch these images " \
                     "into a rotating GIF.\n\nE.g. " \
                     "http://blog.gregzaal.com/2015/08/06/making-an-optimized-gif-in-gimp/"
        rotate_zip.writestr("_README.txt", readme_str)

print("LOGO GENERATION COMPLETE")
