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
    1.28  # How much bigger than the globe the iris clip should be.
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
orthographic = ccrs.Orthographic(central_longitude=-30, central_latitude=22.9)
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
ax.add_feature(LAND)

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
artwork_dict = OrderedDict.fromkeys(["background", "glow", "sea", "land",])
# The elements that will just be referenced by artwork elements.
defs_dict = {}

# BACKGROUND
artwork_dict["background"] = ET.Element(
    "rect",
    attrib={
        "height": "100%",
        "width": "100%",
        "style": "fill:url(#background_gradient);",
    },
)
background_gradient = ET.Element(
    "linearGradient", attrib={"y1": "0%", "y2": "100%",},
)
background_gradient.append(
    ET.Element("stop", attrib={"offset": "0", "style": "stop-color:#13385d",},)
)
background_gradient.append(
    ET.Element("stop", attrib={"offset": "1", "style": "stop-color:#272b2c",},)
)
defs_dict["background_gradient"] = background_gradient

# LAND
mpl_land = root_logo.find(".//svg:g[@id='figure_1']", namespaces)
root_logo.remove(mpl_land)
land_paths = mpl_land.find(".//svg:g[@id='PathCollection_1']", namespaces)
for path in land_paths:
    path.attrib.pop("clip-path")
    path.attrib.pop("style")
land_paths.tag = "clipPath"
defs_dict["land_clip"] = land_paths

artwork_dict["land"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": "50%",
        "style": "fill:url(#land_gradient)",
        "clip-path": "url(#land_clip)",
    },
)
land_gradient = ET.Element("radialGradient")
land_gradient.append(
    ET.Element("stop", attrib={"offset": "0", "style": "stop-color:#d5e488"},)
)
land_gradient.append(
    ET.Element("stop", attrib={"offset": "1", "style": "stop-color:#aec928",},)
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
        "style": "fill:url(#sea_gradient)",
    },
)
sea_gradient = ET.Element("radialGradient")
sea_gradient.append(
    ET.Element("stop", attrib={"offset": "0", "style": "stop-color:#20b0ea"},)
)
sea_gradient.append(
    ET.Element("stop", attrib={"offset": "1", "style": "stop-color:#156475",},)
)
defs_dict["sea_gradient"] = sea_gradient

# GLOW
artwork_dict["glow"] = ET.Element(
    "circle",
    attrib={
        "cx": "50%",
        "cy": "50%",
        "r": f"{52 / CLIP_GLOBE_RATIO}%",
        "style": "fill:url(#glow_gradient);stroke:#ffffff;stroke-width:2;stroke-opacity:0.797414;filter:url(#glow_blur)",
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
            "style": "stop-color:#0aaea7; stop-opacity:0.85882354",
        },
    )
)
glow_gradient.append(
    ET.Element(
        "stop",
        attrib={
            "offset": "0.67322218",
            "style": "stop-color:#18685d; stop-opacity:0.74117649",
        },
    )
)
glow_gradient.append(
    ET.Element(
        "stop", attrib={"offset": "1", "style": "stop-color:#b6df34;",},
    )
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
