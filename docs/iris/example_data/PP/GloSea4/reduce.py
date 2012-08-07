
import glob
import iris
import os

remove = ["prodf_op_sfc_cam_11_20110718_001.pp",
		  "prodf_op_sfc_cam_11_20110719_002.pp",
		  "prodf_op_sfc_cam_11_20110719_003.pp",
		  "prodf_op_sfc_cam_11_20110720_004.pp",
		  "prodf_op_sfc_cam_11_20110720_005.pp",
		  "prodf_op_sfc_cam_11_20110722_008.pp",
		  "prodf_op_sfc_cam_11_20110722_009.pp",
		  "prodf_op_sfc_cam_11_20110722_010.pp",
		  "prodf_op_sfc_cam_11_20110723_011.pp",
		  "prodf_op_sfc_cam_11_20110724_012.pp"]
		  
for f in remove:
	os.remove(f)

fnames = glob.glob("*.pp")
for fname in fnames:

	pps = iris.fileformats.pp.load(fname)
	
	out_fname = fname.rsplit(".")[0] + "_subset.pp"
	outfile = open(out_fname, "wb")

	for pp in pps:	
		if pp.lbuser[3] in [24, 5216]:
			pp.save(outfile)
			
	outfile.close()
	
	os.remove(fname)
