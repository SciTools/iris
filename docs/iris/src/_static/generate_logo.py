from collections import OrderedDict
from copy import deepcopy
from os import system
from pathlib import Path
from PIL.ImageFont import truetype
from re import sub as re_sub
from tempfile import TemporaryFile
from xml.etree import ElementTree as ET
from xml.dom import minidom

from cartopy import crs as ccrs, __version__ as cartopy_version
from cartopy.feature import LAND
from matplotlib import pyplot as plt, rcParams

################################################################################
# Configuration
CLIP_GLOBE_RATIO = (
    1.25  # How much bigger than the globe the iris clip should be.
)
LOGO_PIXELS = 1024  # Pixel size of the square logo.
BANNER_PIXELS = 256  # Pixel height of text banner.
BANNER_TEXT = "Iris"  # Text printed in the banner.
TEXT_GLOBE_RATIO = (
    0.6  # How much smaller than the globe the banner text should be.
)
WRITE_DIRECTORY = Path("")
FILENAME_PREFIX = "iris"  # Start of all filenames to be written.

# The logo's SVG elements can be configured at their point of definition below.

################################################################################
# Plot land using Cartopy.

# Set up orthographic plot.
# figure_inches doesn't influence the final size, but does influence the coastline definition.
figure_inches = 10
fig = plt.figure(0, figsize=(figure_inches,) * 2)
orthographic = ccrs.Orthographic(central_longitude=-30, central_latitude=20)
ax = plt.axes(projection=orthographic)
# TODO: remove version check Iris is pinned appropriately.
if cartopy_version < "0.18":
    ax.outline_patch.set_edgecolor("None")
    ax.background_patch.set_facecolor("None")
else:
    ax.spines["geo"].set_visible(False)


# Establish some sizes and ratios.
# Set plot to include space for the mask to be added.
mpl_points_per_inch = 72

globe_inches = figure_inches / CLIP_GLOBE_RATIO
pad_inches = (figure_inches - globe_inches) / 2
background_points = figure_inches * mpl_points_per_inch
globe_points = globe_inches * mpl_points_per_inch

mpl_font_points = rcParams["font.size"]
inch_font_ratio = mpl_points_per_inch / mpl_font_points
# TODO: remove version check Iris is pinned appropriately.
if cartopy_version < "0.18":
    fig.canvas.draw()
plt.tight_layout(pad=pad_inches * inch_font_ratio)


# Add land with simplified coastlines.
simple_geometries = [
    geometry.simplify(1.0, True) for geometry in LAND.geometries()
]
LAND.geometries = lambda: iter(simple_geometries)
ax.add_feature(LAND, facecolor="#bad92e")

################################################################################
# Matplotlib SVG content.

with TemporaryFile() as temp_svg:
    plt.savefig(temp_svg, format="svg", transparent=True)
    temp_svg.seek(0)

    # Read saved SVG file.
    namespaces = {"svg": "http://www.w3.org/2000/svg"}
    ET.register_namespace("", namespaces["svg"])
    tree = ET.parse(temp_svg)

root_original = tree.getroot()
root_logo = deepcopy(root_original)

################################################################################
# Create new SVG elements for logo.

# The elements that are displayed (not just referenced).
# Order is important for layering.
artwork_dict = OrderedDict.fromkeys(
    ["background", "glow", "haze", "sea", "land", "sun",]
)
# The elements that will just be referenced by artwork elements.
defs_dict = {}

# LAND
land = root_logo.find("svg:g", namespaces)
root_logo.remove(land)
artwork_dict["land"] = land

# SEA
# Not using Cartopy for sea since it doesn't actually render curves/circles.
artwork_dict["sea"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": f"{50.5 / CLIP_GLOBE_RATIO}%",
        "style": "fill:#0078b0;",
    },
)

# BACKGROUND
artwork_dict["background"] = ET.Element(
    "rect",
    attrib={"height": "100%", "width": "100%", "style": "fill:#333333;"},
)

# GLOW
artwork_dict["glow"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": f"{52.5 / CLIP_GLOBE_RATIO}%",
        "style": "fill:#cadc6c; fill-opacity:1; filter:url(#glow_blur)",
    },
)
glow_blur = ET.Element("filter")
glow_blur.append(ET.Element("feGaussianBlur", attrib={"stdDeviation": "12"}))
defs_dict["glow_blur"] = glow_blur

# HAZE
artwork_dict["haze"] = ET.Element(
    "rect",
    attrib={
        "height": "100%",
        "width": "100%",
        "style": "fill:url(#haze_gradient);",
    },
)
haze_gradient = ET.Element(
    "linearGradient",
    attrib={"x1": "50%", "y1": "0%", "x2": "50%", "y2": "100%",},
)
haze_gradient.append(
    ET.Element(
        "stop",
        attrib={
            "offset": "0",
            "style": "stop-color:#003f7d; stop-opacity:0.7",
        },
    )
)
haze_gradient.append(
    ET.Element(
        "stop",
        attrib={"offset": "1", "style": "stop-color:#003f7d; stop-opacity:0",},
    )
)
defs_dict["haze_gradient"] = haze_gradient

# SUN
artwork_dict["sun"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": f"{50.5 / CLIP_GLOBE_RATIO}%",
        "style": "fill:url(#sun_gradient);",
    },
)
sun_gradient = ET.Element(
    "radialGradient",
    attrib={"cx": "50%", "cy": "50%", "fx": "50%", "fy": "50%", "r": "50%"},
)
sun_gradient.append(
    ET.Element(
        "stop",
        attrib={"offset": "0", "style": "stop-color:white; stop-opacity:0.5"},
    )
)
sun_gradient.append(
    ET.Element(
        "stop",
        attrib={
            "offset": "0.75",
            "style": "stop-color:white; stop-opacity:0.0",
        },
    )
)
defs_dict["sun_gradient"] = sun_gradient

# CLIP
# SVG string representing bezier curves, originally drawn on a 100x100pt canvas.
clip_scaling = background_points / 100
clip_string = " ".join(
    [
        "m 49.156375, 0.849600",
        "c -2.004127,0.022656 -4.052215,0.199656 -6.131272,0.699334 "
        "-12.265137,2.767430 -9.849795,37.691796 2.644530,47.261832 "
        "2.051082,1.568532 4.659645,-9.571056 17.075940,-17.948225 "
        "9.784855,-6.603828 18.734193,-5.864506 19.893308,-8.681892 "
        "1.132942,-2.747408 -14.109514,-21.569843 -33.483008,-21.352034",
        "z",
        "m -18.576640,10.694000",
        "c -6.077321,0.299711 -28.650425,11.042676 -30.550749,31.556523 "
        "-1.159916,12.520272 32.803942,21.016357 45.766328,12.090686 "
        "2.128010,-1.468619 -7.662840,-7.393078 -11.793578,-21.786633 "
        "-3.254959,-11.346408 0.209795,-19.629668 -2.108030,-21.602808 "
        "-0.229788,-0.199939 -0.689354,-0.299739 -1.314770,-0.299739",
        "z",
        "m 56.247734,14.108826",
        "c -12.087004,-0.199939 -31.334318,9.960682 -35.264645,21.092283 "
        "-0.859198,2.437729 10.541548,1.478616 22.346117,10.694991 "
        "9.303307,7.263230 11.366977,16.005048 14.403240,16.237838 "
        "3.272940,0.199939 18.990947,-24.507108 7.382101,-44.052836 "
        "-1.605500,-2.697480 -4.837481,-3.916344 -8.866714,-3.976270",
        "z",
        "m -31.306644,26.163545",
        "c -2.583585,-0.064853 1.855269,10.483186 -3.264948,24.559047 "
        "-4.035229,11.092632 -11.709756,15.756257 -10.992527,18.715499 "
        "0.769282,3.187020 29.176034,10.489190 44.177614,-6.593831 "
        "8.297245,-9.441180 -14.186743,-36.284094 -29.919938,-36.682726",
        "z",
        "m -4.555742,4.725588",
        "c -2.676498,0.199656 -10.815691,5.424923 -23.446187,4.985340 "
        "-11.796975,-0.399624 -18.603514,-6.264157 -21.196391,-4.665663 "
        "-2.795388,1.718401 -0.959104,30.989047 19.920383,39.978664 "
        "11.548507,4.975343 30.124147,-24.704924 25.640539,-39.790818 "
        "-0.119879,-0.399624 -0.419609,-0.499536 -0.919143,-0.499536",
        "z",
    ]
)

# Scale the clip path.
scale_func = lambda match: "{:6f}".format(float(match.group(0)) * clip_scaling)
clip_string = re_sub(r"\d*\.\d*", scale_func, clip_string)

iris_clip = ET.Element("clipPath")
iris_clip.append(ET.Element("path", attrib={"d": clip_string}))
defs_dict["iris_clip"] = iris_clip

################################################################################
# Create logo SVG

for dim in ("width", "height"):
    root_logo.attrib[dim] = str(LOGO_PIXELS)

# Move root contents into a logo subgroup (text will be stored separately).
logo_group = ET.Element("svg", attrib={"id": "logo_group"})
logo_group.extend(list(root_logo))
[root_logo.remove(child) for child in list(root_logo)]
root_logo.append(logo_group)
# Also transfer viewBox attribute.
logo_group.attrib["viewBox"] = root_logo.attrib.pop("viewBox")


def populate_element_group(group, children_dict):
    for name, element in children_dict.items():
        element.attrib["id"] = name
        group.append(element)
    logo_group.append(group)


defs_element = ET.Element("defs")
populate_element_group(defs_element, defs_dict)

artwork_element = ET.Element("g", attrib={"clip-path": "url(#iris_clip)"})
populate_element_group(artwork_element, artwork_dict)

################################################################################
# Create Banner SVG.

root_banner = deepcopy(root_logo)
root_banner.attrib["height"] = str(BANNER_PIXELS)

text_font = "georgia"
text_size = BANNER_PIXELS * TEXT_GLOBE_RATIO
text_width = truetype(text_font, int(text_size)).getsize(BANNER_TEXT)[0]
text_width *= 1.35  # Hack adjustment.
text_buffer = text_width * 0.05

text_x = BANNER_PIXELS + (text_buffer / 2)
# Manual y centring since SVG dominant-baseline not widely supported.
text_y = BANNER_PIXELS - (BANNER_PIXELS - text_size) / 2
text_y *= 0.975  # Slight offset

# Resize image to accommodate text.
root_banner.attrib["width"] = str(BANNER_PIXELS + text_buffer + text_width)
# Left-align the logo.
banner_logo_group = root_banner.find("svg", namespaces)
banner_logo_group.attrib["preserveAspectRatio"] = "xMinYMin meet"

text = ET.Element(
    "text",
    attrib={
        "x": str(text_x),
        "y": str(text_y),
        "style": f"font-size:{text_size}pt; font-family:{text_font}",
    },
)
text.text = BANNER_TEXT
root_banner.append(text)

################################################################################
# Write files.


def write_svg_file(svg_root, filename_suffix):
    input_string = ET.tostring(svg_root)
    pretty_xml = minidom.parseString(input_string).toprettyxml()
    # Remove extra empty lines from Matplotlib.
    pretty_xml = "\n".join(
        [line for line in pretty_xml.split("\n") if line.strip()]
    )

    filename = f"{FILENAME_PREFIX}-{filename_suffix}.svg"
    write_path = WRITE_DIRECTORY.joinpath(filename)
    with open(write_path, "w") as f:
        f.write(pretty_xml)
    return write_path


path_logo = write_svg_file(root_logo, "logo")
path_banner = write_svg_file(root_banner, "logo-title")

path_logo_ico = path_logo.with_suffix(".ico")
system(f"convert -resize 16x16 {path_logo} {path_logo_ico}")
