import sys
import itertools
import operator
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import fastcluster

def getdata(file):
    for line in file:
        g = line.strip().split('\t')
        yield g

def cluster(loc_time_data):
    mat = np.array(loc_time_data)
    mat2 = mat[0:len(mat),1:3]
    try:
        A = np.array(mat2)
        
        d = sch.distance.pdist(A)
        d[np.abs(d) < 1e-11] = 0
        #Z = sch.linkage(d,method='average')
        Z = fastcluster.linkage(d,method="average")
        T = sch.fcluster(Z, 10, 'distance')
        #print cuid,T
        
        u,indices = np.unique(T, return_index=True)
        #print indices
        for index in indices:
            print '%s\t%s'%(cuid,'\t'.join(loc_time_data[index]))
    except Exception , e:
        return
    
for cuid,data in itertools.groupby(getdata(sys.stdin),\
        operator.itemgetter(0)):
    
    loc_time_data = []
    num = 0 
    for info in data:
        if num > 1000:
            cluster(loc_time_data)
            loc_time_data = []
            num = 0
        
        time = info[1]
        x = info[2]
        y = info[3]
        if len(x) < 4 or len(y) < 4:
            continue
        loc_time_data.append([time,x,y])
        num += 1 
    
    if len(loc_time_data) == 1:
        print '%s\t%s'%(cuid,'\t'.join(loc_time_data[0]))
        continue
    elif len(loc_time_data) == 0:
        continue
    else:
        cluster(loc_time_data)

