import iris


def reduce(pp, factor):
	pp.data = pp.data[::factor, ::factor]
	pp.x = pp.x[::factor]
	pp.y = pp.y[::factor]
	pp.x_lower_bound = pp.x_lower_bound[::factor]
	pp.x_upper_bound = pp.x_upper_bound[::factor]
	pp.y_lower_bound = pp.y_lower_bound[::factor]
	pp.y_upper_bound = pp.y_upper_bound[::factor]
	pp.lbnpt = pp.data.shape[1]
	pp.lbrow = pp.data.shape[0]
	pp.bdx *= factor
	pp.bdy *= factor



pps = list(iris.fileformats.pp.load("theta_and_orog_subset.pp"))

times = ['2009-09-09 17:10:00', '2009-09-09 17:40:00']

with open("theta_and_orog_subset_b.pp", "wb") as outfile:
	for pp in pps:
		if str(pp.t1) in times:
			if ((pp.lblev%2)==0 and pp.lblev<=16) or pp.lbuser[3]==33:
				reduce(pp, 2)
				pp.save(outfile)




pps = list(iris.fileformats.pp.load("air_potential_and_air_pressure.pp"))

times = ['2009-09-09 22:10:00', '2009-09-09 22:40:00']
levs = [1, 5, 9]

pps = [pp for pp in pps if str(pp.t1) in times]
pps = [pp for pp in pps if pp.lblev in levs]

with open("air_potential_and_air_pressure_subset_b.pp", "wb") as outfile:
	for pp in pps:
		reduce(pp, 4)
		pp.save(outfile)


