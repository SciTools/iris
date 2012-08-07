
import iris

theta = list(iris.fileformats.pp.load("theta.pp"))[::5]
u_wind = list(iris.fileformats.pp.load("u_wind.pp"))[::5]
v_wind = list(iris.fileformats.pp.load("v_wind.pp"))[::5]

def halve(pp):
	pp.data = pp.data[::2, ::2]
	pp.bdx *= 2
	pp.bdy *= 2
	pp.lbnpt = pp.data.shape[1]
	pp.lbrow = pp.data.shape[0]

with open("theta_subset.pp", "wb") as outfile:
	for pp in theta:
		halve(pp)
		pp.save(outfile)

with open("u_wind_subset.pp", "wb") as outfile:
	for pp in u_wind:
		halve(pp)
		pp.save(outfile)

with open("v_wind_subset.pp", "wb") as outfile:
	for pp in v_wind:
		halve(pp)
		pp.save(outfile)
