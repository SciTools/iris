import numpy as np
import iris.tests
import iris.tests.stock as istk
from iris.coords import AuxCoord, DimCoord
import iris._lazy_data as lazy

cube = istk.simple_pp()
co = cube.coord("latitude")
co_b = co.copy()
co_b.guess_bounds()


def test(co):
    # print(f'coord shape: {co.shape}')
    print("------------REPR:-------")
    print(repr(co))
    print("------------STR:--------")
    print(str(co))
    print("------------------------")
    print("")


print("Small section of latitude coord[:2]")
test(co[:2])

with np.printoptions(precision=3):
    co2 = co_b.copy()
    multibounds = np.concatenate([co2.bounds, co2.bounds], axis=-1)
    co2.bounds = multibounds
    print("co[:2] with 4x bounds + printoptions(precision=3) ")
    test(co2[:2])

print("\nLonger section co[:5]")
test(co[:5])

print("\nLonger section co[:5] with bounds")
test(co_b[:5])

print("\nShort co[:1] -->  one-line bounds")
test(co_b[:1])

print("\nStill longer co[-15:] with bounds")
test(co_b[-15:])

print("\nFull co")
test(co)

print("\nFull co with bounds")
test(co_b)

print("\nSmall co with lazy points")
co2 = AuxCoord.from_coord(co[:4])
co2.points = lazy.as_lazy_data(co2.points)
assert co2.has_lazy_points()
assert not co2.has_lazy_bounds()
print(co2.core_points())
test(co2)

print("\nSmall co with real points but lazy bounds")
co2.points = lazy.as_concrete_data(co2.points)
co2.guess_bounds()
co2.bounds = lazy.as_lazy_data(co2.bounds)
assert not co2.has_lazy_points()
assert co2.has_lazy_bounds()
test(co2)

print("\nFloat with masked points")
co2 = AuxCoord.from_coord(co[:10])
pts = co2.points
pts = np.ma.masked_array(pts)
pts[[1, 3]] = np.ma.masked
co2.points = pts
test(co2)

print("\nInteger")
co = AuxCoord(range(5), long_name="integer_points")
test(co)

print("\nInteger with masked points")
pts = co.points
pts = np.ma.masked_array(pts)
pts[[1, 3]] = np.ma.masked
co.points = pts
test(co)

print("\nLonger integers with masked points")
pts = np.ma.arange(995, 1020)
pts[[1, 7]] = np.ma.masked
co = AuxCoord(pts, long_name="integer_points", var_name="qq")
test(co)

print("\ndates with bounds")
co = AuxCoord(
    np.linspace(0, 100, 10),
    units="days since 2015-05-17",
    var_name="x",
    attributes={"a": 14, "b": None},
)
co.guess_bounds()
test(co)

print("\ndates with masked points")
pts = co.points
pts = np.ma.masked_array(pts)
pts[[2, 5]] = np.ma.masked
co.points = pts
test(co)

print("\nmultidimensional")
bigdata = np.exp(np.random.uniform(-3, 6, size=(7, 5, 4)))
co = AuxCoord(bigdata)
test(co)

print("\nsmall multidim")
test(co[0, :2, :2])

print("")
print("SCALAR:")
test(AuxCoord([15.0], standard_name="forecast_period", units="hours"))

print("")
print("==== REAL DATA nc test cube coords ...")
test_ncfile = iris.tests.get_data_path(
    "NetCDF/global/xyt/SMALL_hires_wind_u_for_ipcc4.nc"
)
for coord in iris.load_cube(test_ncfile)[:4].coords():
    test(coord)
