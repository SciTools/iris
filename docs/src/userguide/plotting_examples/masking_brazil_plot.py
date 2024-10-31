"""Global cube masked to Brazil and plotted with quickplot."""

import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt
from iris.util import mask_cube_from_shapefile

country_shp_reader = shpreader.Reader(
    shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
)
brazil_shp = [
    country.geometry
    for country in country_shp_reader.records()
    if "Brazil" in country.attributes["NAME_LONG"]
][0]

cube = iris.load_cube(iris.sample_data_path("air_temp.pp"))
brazil_cube = mask_cube_from_shapefile(cube, brazil_shp)

qplt.pcolormesh(brazil_cube)
plt.show()
