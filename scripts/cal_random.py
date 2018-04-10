#!/usr/bin/env python
import numpy as np
import sys
import operator
import matplotlib.pyplot as plt
fn=sys.argv[1]
linedata=[]
alldata=[]
skip=1
with open(fn) as f:
    for line in f:
        linedata=[]
        for x in line.split():
            try:
                linedata.append(float(x.split(":")[1]))
            except:
                pass
        linedata=np.array(linedata)
        alldata.append(linedata)
f.close()


outfn=fn+"_norm_random"
print ('write to ', outfn)
xind=range(len(alldata))
xind=np.array(xind)
np.random.shuffle(xind)
#print(xind)
f=open(outfn,'w')
for x in xind:
    f.write(str(x)+'\n')
f.close()
