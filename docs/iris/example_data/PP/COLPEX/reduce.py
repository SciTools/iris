import iris


#theta = iris.load_strict("theta_and_orog_subset.pp", 'air_potential_temperature')
#print theta
#print theta.shape
#if theta.shape == (6,70,100,100):
#	theta = theta[::3, ::10, :, :]
#	iris.save(theta, "theta_and_orog_subset__.pp")


pps = list(iris.fileformats.pp.load("theta_and_orog_subset.pp"))

times = ['2009-09-09 17:10:00', '2009-09-09 17:40:00']

with open("theta_and_orog_subset_b.pp", "wb") as outfile:
	for pp in pps:
		if str(pp.t1) in times:
#			if (pp.lblev%10)==0 or pp.lbuser[3]==33:
			if ((pp.lblev%2)==0 and pp.lblev<=16) or pp.lbuser[3]==33:
				pp.save(outfile)



pps = list(iris.fileformats.pp.load("air_potential_and_air_pressure.pp"))

times = ['2009-09-09 22:10:00', '2009-09-09 22:40:00']
levs = [1, 5, 9]

pps = [pp for pp in pps if str(pp.t1) in times]
pps = [pp for pp in pps if pp.lblev in levs]

with open("air_potential_and_air_pressure_subset_b.pp", "wb") as outfile:
	for pp in pps:
		pp.data = pp.data[::4, ::4]
		pp.x = pp.x[::4]
		pp.y = pp.y[::4]
		pp.x_lower_bound = pp.x_lower_bound[::4]
		pp.x_upper_bound = pp.x_upper_bound[::4]
		pp.y_lower_bound = pp.y_lower_bound[::4]
		pp.y_upper_bound = pp.y_upper_bound[::4]
		pp.lbnpt /= 4
		pp.lbrow /= 4
		pp.bdx *= 4
		pp.bdy *= 4
		pp.save(outfile)
