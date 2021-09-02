Examples generated from CDL example sections in UGRID conventions v1.0 
  ( see webpage: https://ugrid-conventions.github.io/ugrid-conventions/ )

CHANGES:
    * all files had a data-var added, for ease of iris-roundtripping
    * EX4 :
        - had a couple of missing ";"s at lineends
        - existing var "Mesh2_surface" is tied to the node dimension but has ':location = "face"'
        - actually, ALL the formula terms should map to 'face', not nodes.
        - created data-var maps (layers, faces), as reqd

/home/h05/itpp/git/iris/iris_main/lib/iris/tests/results/ugrid_ref
    $ for f in $(ls *.cdl); do n=$(echo $f | grep -o "[^.]*" | grep "ugrid"); echo $n; ncgen $n.cdl -4 -o $n.nc; done
    ugrid_ex1_1d_mesh
    ugrid_ex2_2d_triangular
    ugrid_ex3_2d_flexible
    ugrid_ex4_3d_layered

    (ncdump -h $n.nc >$n.ncdump.txt)
    $ for f in $(ls *.cdl); do n=$(echo $f | grep -o "[^.]*" | grep "ugrid"); echo $n; ncdump -h $n.nc >$n.ncdump.txt; done

    (xxdiff $n.cdl $n.ncdump.txt;)
    $ for f in $(ls *.cdl); do n=$(echo $f | grep -o "[^.]*" | grep "ugrid"); echo $n; xxdiff $n.cdl $n.ncdump.txt; done

Then for compatibility testing...
    (python iris_loadsave.py $n.nc)
    $ for f in $(ls *.cdl); do n=$(echo $f | grep -o "[^.]*" | grep "ugrid"); echo $n; python iris_loadsave.py $n.nc; done



So each of (4) examples has :
    <ex>.cdl            : original text from UGRID webpage
    <ex>.nc             : "ncgen" output  = generated netcdf file
    <ex>.ncdump.txt     : "ncdump -h" output  = re-generated CDL from nc file
(from iris_loadsave.py)
    <ex>_REDUMP_cdl.txt             : same as .ncdump.txt
    <ex>_RESAVED.nc                 : from loading <ex>.nc + re-saving it
    <ex>_RESAVED_REDUMP_cdl.txt     : ncdump of _RESAVED.nc

NEWSTYLE CHECKS ?
<ex>.CDL :
  ==> ncgen ==> <ex>.nc
  ==> load ==> ex_iris_data
  ==> save ==> ex_iris_resaved
  ==> load ==> ex_iris_saveload :: COMPARE with ex_iris_data

