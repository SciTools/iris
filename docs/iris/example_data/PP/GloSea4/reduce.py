
import glob
import iris
import os

fnames = glob.glob("*.pp")
for fname in fnames:

	
#	# t0, param24
#	p24t0 = [pp for pp in pps if pp.lbuser[3] == 24]
#	p24t0 = [pp for pp in pps if str(pp.t1) == '2012-01-01 00:00:00']
#	
#	out_fname = fname.rsplit(".")[0] + "_p24t0subset.pp"
#	outfile = open(out_fname, "wb")
#
#	for pp in p24t0:
#		pp.save(outfile)
#			
#	outfile.close()

	pps = iris.fileformats.pp.load(fname)

	# reduce xy down to 10x10
	out_fname = fname.rsplit(".")[0] + "_subset.pp"
	outfile = open(out_fname, "wb")

	for pp in pps:
		if pp.lbuser[3] == 24:
			pp.data = pp.data[::2, ::5]
			pp.bdy *= 2
			pp.bdx *= 5
			pp.lbnpt = pp.data.shape[1]
			pp.lbrow = pp.data.shape[0]
			pp.save(outfile)
			
	outfile.close()


	
	os.remove(fname)


