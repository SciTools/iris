
import iris

theta = list(iris.fileformats.pp.load("theta.pp"))[::5]
u_wind = list(iris.fileformats.pp.load("u_wind.pp"))[::5]
v_wind = list(iris.fileformats.pp.load("v_wind.pp"))[::5]

with open("theta_subset.pp", "wb") as outfile:
	for pp in theta:
		pp.save(outfile)

with open("u_wind_subset.pp", "wb") as outfile:
	for pp in u_wind:
		pp.save(outfile)

with open("v_wind_subset.pp", "wb") as outfile:
	for pp in v_wind:
		pp.save(outfile)
