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
from re import sub as re_sub
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

# Pixel size of the square logo.
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

# Establish some sizes and ratios - allows space for the mask to be added after plotting.
# figure_inches doesn't influence the final size, but does influence the coastline definition.
figure_inches = 10
mpl_points_per_inch = 72

globe_pad = (1 - (1 / CLIP_GLOBE_RATIO)) / 2
background_points = figure_inches * mpl_points_per_inch

# XML ElementTree setup.
namespaces = {"svg": "http://www.w3.org/2000/svg"}
ET.register_namespace("", namespaces["svg"])

################################################################################
# Create new SVG elements for logo.

# The elements that are displayed (not just referenced).
# Order is important for layering.
artwork_dict = dict.fromkeys(["background", "glow", "sea", "land",])
# The elements that will just be referenced by artwork elements.
defs_dict = {}

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
    ET.Element("stop", attrib={"offset": "1", "stop-color": "#272b2c",},)
)
defs_dict["background_gradient"] = background_gradient

# LAND
# (Using Matplotlib and Cartopy).

# Create land with simplified coastlines.
simple_geometries = [
    geometry.simplify(1.0, True) for geometry in LAND.geometries()
]
LAND.geometries = lambda: iter(simple_geometries)

# Variable that will store the sequence of land-shaped SVG clips for each longitude.
land_clips = []

# Create a sequence of longitude values.
central_longitude = -30
central_latitude = 22.9
rotation_frames = 180
rotation_longitudes = np.linspace(start=central_longitude + 360,
                                  stop=central_longitude,
                                  num=rotation_frames,
                                  endpoint=False)
# Normalise to -180..+180
rotation_longitudes = (rotation_longitudes + 360.0 + 180.0) % 360.0 - 180.0

for lon in rotation_longitudes:
    # Use Matplotlib and Cartopy to generate land-shaped SVG clips for each longitude.

    projection_rotated = ccrs.Orthographic(central_longitude=lon,
                                           central_latitude=central_latitude)

    # Use constants set earlier to achieve desired dimensions.
    fig = plt.figure(0, figsize=(figure_inches,) * 2)
    ax = plt.subplot(projection=projection_rotated)
    plt.subplots_adjust(left=globe_pad, bottom=globe_pad, right=1 - globe_pad, top=1 - globe_pad)
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
        path.attrib = {"d": path.attrib["d"]}
    land_paths.tag = "clipPath"

    land_clips.append(land_paths)

# Extract the final land clip for use as the default.
defs_dict["land_clip"] = land_clips[-1]

artwork_dict["land"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": "50%",
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

# SEA
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

# GLOW
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
glow_blur.append(ET.Element("feGaussianBlur", attrib={"stdDeviation": "10"}))
defs_dict["glow_blur"] = glow_blur

# CLIP
# SVG string representing bezier curves, originally drawn on a 100x100 canvas.
clip_scaling = background_points / 100
clip_string = " ".join(
    [
        "m 48.715792,0.06648723",
        "c -3.695266,-0.02152 -6.340628,0.83101 -7.803951,1.85111997 "
        "-7.780711,5.424073 -6.020732,25.7134678 -3.574593,32.4721008 "
        "2.886671,7.975828 7.326212,15.759187 9.421487,14.669421 "
        "2.708508,-1.024559 5.323978,-11.805637 17.531882,-19.077586 "
        "9.504775,-5.661737 16.925959,-5.68594 17.734968,-7.414812 "
        "0.402381,-1.120006 0.212418,-1.845229 -0.645975,-3.096052",
        "C 80.521209,18.219863 75.12671,9.4573304 65.13349,4.5345992 "
        "58.440375,1.2375202 52.903757,0.09084923 48.715792,0.06648723",
        "Z",
        "M 31.075381,11.520477",
        "c -0.433583,-0.0085 -0.926497,0.09327 -1.552935,0.288374",
        "C 28.090574,12.254804 18.250845,14.778908 10.571472,22.949149 "
        "0.88982409,33.249658 -0.41940991,41.930793 0.64560611,45.630785 "
        "3.3001452,54.852655 22.76784,59.415505 29.827347,59.131763 "
        "38.158213,58.796931 46.788636,56.896487 46.419722,54.52773 "
        "46.301117,51.584413 37.054716,45.716497 34.044816,31.630083 "
        "31.701409,20.66275 33.97219,13.457909 32.609708,12.139069 "
        "32.092072,11.724892 31.632837,11.531455 31.075381,11.520477",
        "Z",
        "m 55.776453,13.86169",
        "c -8.577967,-0.02376 -19.956997,5.29967 -24.117771,8.645276 "
        "-6.546803,5.264163 -12.433634,11.974719 -10.769772,13.669965 "
        "1.792557,2.310114 12.655507,1.515124 23.210303,11.10723 "
        "8.217686,7.468185 10.533621,14.657837 12.396084,14.90817 "
        "1.16895,0.04416 1.786961,-0.363913 2.688309,-1.582904 "
        "0.90134,-1.21901 7.406651,-9.158433 8.90987,-20.371091 "
        "1.895163,-14.136193 -2.04942,-21.944169 -5.04381,-24.299175 "
        "-1.865818,-1.467402 -4.41389,-2.069555 -7.273213,-2.077471",
        "z",
        "M 56.128531,51.700411",
        "c -0.639249,0.02574 -1.086221,0.223499 -1.286275,0.624277 "
        "-1.600644,2.452282 2.497685,12.741825 -3.187004,25.942017 "
        "-4.425963,10.277335 -10.415712,14.745486 -10.073661,16.629076 "
        "0.320069,1.14727 0.891984,1.619829 2.307444,2.117259 "
        "1.415458,0.49743 10.830701,4.35328 21.752971,2.34619 "
        "13.770075,-2.530379 19.832746,-8.768519 21.103889,-12.400221 "
        "3.168142,-9.051698 -9.950654,-24.412078 -15.825463,-28.413798 "
        "-5.63292,-3.836942 -12.021817,-6.956398 -14.791901,-6.8448",
        "z",
        "m -5.299096,4.785932",
        "C 47.570588,56.612385 39.424575,62.816068 26.2357,61.471757 "
        "15.282622,60.3553 9.2649892,55.927072 7.6139222,56.840861",
        "c -0.9711391,0.664932 -1.235964,1.365498 -1.2625019,2.891917 "
        "-0.02655,1.526442 -0.7128592,11.84897 4.5342547,21.821195 "
        "6.61521,12.572318 14.306917,16.524769 18.086914,16.635267 "
        "9.421278,0.275342 19.693467,-17.19462 21.610339,-24.128637 "
        "2.262092,-8.182753 3.156328,-17.140105 0.833056,-17.514311 "
        "-0.173865,-0.04967 -0.369295,-0.06832 -0.586549,-0.05995",
        "z",
    ]
)


# Scale the clip path.
def scale_func(match):
    scaled = float(match.group(0)) * clip_scaling
    return "{:6f}".format(scaled)


clip_string = re_sub(r"\d*\.\d*", scale_func, clip_string)
iris_clip = ET.Element("clipPath")
iris_clip.append(ET.Element("path", attrib={"d": clip_string}))
defs_dict["iris_clip"] = iris_clip

################################################################################
# Create SVG's


def svg_logo(defs_dict, artwork_dict):
    # Group contents into a logo subgroup (so text can be stored separately).
    logo_group = ET.Element("svg", attrib={"id": "logo_group"})
    logo_group.attrib["viewBox"] = " ".join(
        ["0"] * 2 + [str(background_points)] * 2
    )

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
    for dim in ("width", "height"):
        root.attrib[dim] = str(LOGO_PIXELS)
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