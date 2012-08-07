
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

;		dots = STRSPLIT(fname, '.', extract=0)
;		last_dot = dots[size(dots, /n_el)-1]
;		fname = STRMID(fname, 0, last_dot-1) + "_subset.pp"
;		print, fname
		
		ppw, pps, fname
	END
	
end
