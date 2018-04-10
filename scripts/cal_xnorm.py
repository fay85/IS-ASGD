#!/usr/bin/env python
import numpy as np
import sys
import operator
import matplotlib.pyplot as plt
fn=sys.argv[1]
lr=float(sys.argv[2])
outfn=fn+"_prob_lip_"+sys.argv[2]
linedata=[]
alldata=[]
skip=1
balancing=False
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

f=open(outfn,'w')
norm=0
lip=0
Lipschitz_array=[]
for t in alldata:
    norm=np.linalg.norm(t)
    #lip=2*(norm/np.sqrt(lr)+1)*norm
    lip=norm
    Lipschitz_array.append(lip)
    f.write(str(lip)+'\n')
f.close()
print ('total written ',len(alldata))
print ('write to ', outfn)

avg_Lip=np.mean(Lipschitz_array)
tmp=Lipschitz_array-avg_Lip
eta=np.linalg.norm(tmp)
print('eta ',eta/len(Lipschitz_array))

b=dict(enumerate(Lipschitz_array))
sorted_b=sorted(b, key=b.get)
outfn=fn+"_norm_balanced_"+sys.argv[2]
print ('write to ', outfn)
f=open(outfn,'w')
if balancing:
    for i in range(int(len(sorted_b)/2)):
        f.write(str(sorted_b[i])+'\n')
        f.write(str(sorted_b[len(sorted_b)-1-i])+'\n')
    if len(sorted_b)%2:
        f.write(str(sorted_b[len(sorted_b)/2])+'\n')
else:
    for i in range(len(sorted_b)):
        f.write(str(sorted_b[i])+'\n')    
f.close()
a = np.linalg.norm(Lipschitz_array)
b = np.sum(Lipschitz_array)
n = len(Lipschitz_array)
print (n)
print ((b*b)/(a*a*n))

