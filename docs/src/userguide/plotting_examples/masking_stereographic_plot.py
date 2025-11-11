"""Masking data with a stereographic projection and plotted with quickplot."""

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt
from iris.util import mask_cube_from_shape

# Define WGS84 coordinate reference system
wgs84 = ccrs.PlateCarree(globe=ccrs.Globe(ellipse="WGS84"))

country_shp_reader = shpreader.Reader(
    shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
)
uk_shp = [
    country.geometry
    for country in country_shp_reader.records()
    if "United Kingdom" in country.attributes["NAME_LONG"]
][0]

cube = iris.load_cube(iris.sample_data_path("toa_brightness_stereographic.nc"))
uk_cube = mask_cube_from_shape(cube=cube, shape=uk_shp, shape_crs=wgs84)

plt.figure(figsize=(12, 5))
# Plot #1: original data
ax = plt.subplot(131)
qplt.pcolormesh(cube, vmin=210, vmax=330)
plt.gca().coastlines()
plt.suptitle("Original Data")

# Plot #2: UK geometry
ax = plt.subplot(132, title="Mask Geometry", projection=ccrs.Orthographic(-5, 45))
ax.set_extent([-12, 5, 49, 61])
ax.add_geometries(
    [
        uk_shp,
    ],
    crs=wgs84,
    edgecolor="None",
    facecolor="orange",
)
plt.gca().coastlines()

# Plot #3 masked data
ax = plt.subplot(133, title="Masked Data")
qplt.pcolormesh(uk_cube, vmin=210, vmax=330)
plt.gca().coastlines()
plt.suptitle("Masked Data")

plt.tight_layout()
plt.show()
