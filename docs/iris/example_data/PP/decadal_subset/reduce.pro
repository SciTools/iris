
pro reduce

	spawn, "rm ajnuqa.pmj2*"
	spawn, "rm ajnuqa.pmj3*"
	spawn, "rm ajnuqa.pmj4*"
	spawn, "rm ajnuqa.pmj5*"
	spawn, "rm ajnuqa.pmj6*"
	spawn, "rm ajnuqa.pmj7*"
	spawn, "rm ajnuqa.pmj8*"
	spawn, "rm ajnuqa.pmj9*"
	spawn, "rm ajnuqa.pmk*"

	fnames = FILE_SEARCH("*.pp")

	for i=0, size(fnames,/n_el)-1 DO BEGIN
		fname = fnames[i]
		print, fname
		
		pps = ppa(fname, "f.lbuser[3] eq 3236")
		
		; TODO: We could reduce the res here too

		ppw, pps, fname
	END
	
end
