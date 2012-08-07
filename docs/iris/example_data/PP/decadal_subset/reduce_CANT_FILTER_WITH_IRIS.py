import glob
import iris

fnames = glob.glob("*.pp")
for fname in fnames:
	pps = iris.fileformats.pp.load(fname)
	out_fname = fname.rsplit(".")[0] + "_subset.pp"
	with open(out_fname, "wb") as outfile:
		for pp in pps:
			if pp.lbuser[3] == 3236:
				pp.save(outfile)
