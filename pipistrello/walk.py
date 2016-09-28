#!/usr/bin/env python
import  os

database_dir = ('/home/esp-shared-b/RegCM_Data/CAM2')
files_to_catalogue = [ os.path.join(p,filename) 
                       for (p,d,f) in os.walk(database_dir)
                       for filename in f ]

for i in range(len(files_to_catalogue)):
    print(files_to_catalogue[i])
