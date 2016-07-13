import numpy as np
import util
import sys
import itertools
import operator
from datetime import *
from common import *

import scipy.cluster.hierarchy as hcluster
import scipy.spatial.distance as dist

#dt = np.dtype([('gridid','u4'),('mactag','S12'),('uid','S34'),('wf_list','S162'),('x','i4'),('y','i4'),('t','i4')])
#dt = np.dtype([('gridid','u4'),('mactag','S12'),('uid','S34'),('wf_list','S162'),('x','i4'),('y','i4'),('t','i4')])
dt = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('t','i4')])
dt = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('tm_info','S10')])

#merge same group's wifi
   

def get_largest_dur(data):
    day2dt = {}
    for t in data['t']:
        dt = datetime.fromtimestamp(t)
        day = dt.strftime('%m%d')
        if day not in day2dt:
            day2dt[day] = []
        day2dt[day].append(t)
    dur = []
    for L in day2dt.values():
        dur.append( max(L) - min(L))
    return max(dur)

    
def user_fp_group(data,key,user,filter = 'mid',merge = False,thr = 0.2):
#data = np.fromiter(data,dtype = dt)
    if len(data.shape) == 0 or data.shape[0] == 1:
        print '\t'.join([key,user,'%s' % data['wf_list'],str(data['x']),str(data['y']),'1'])
        return
    dists = get_pdist(data,100)
#print dists
    clusters = hcluster.linkage(dists,method = 'average')
#   print clusters
    r = hcluster.fcluster(clusters,thr,'distance')
    ids = np.unique(r)
    sz = []
    for id in ids:
        sz.append(data[r==id].shape[0])
    
    mid_size = max(1.1,max(sz) / 2.0)
    for id in ids:
        d = data[r==id]
        if filter == 'mid' and d.shape[0] < mid_size:
            continue
        if merge == True:
            print '\t'.join([key,user,wf_to_str(get_mean_wf(d)),str(np.median(d['x'])),str(np.median(d['y'])),str(get_largest_dur(d)),str(d.shape[0])])
            continue
        for od in d:
            print '\t'.join([key,user,od['wf_list'],str(od['x']),str(od['y']),str(od['t']),str(id)])
        #print '\t'.join([key,user,wf_to_str(get_mean_wf(d)),str(np.median(d['x'])),str(np.median(d['y'])),str(get_largest_dur(d)),str(d.shape[0])])

def get_data(file):
    for line in file:
        yield line.strip()

import itertools 
def fp_group(data):
    for grid_mac,data in itertools.groupby(get_data(data),lambda x:x.split('\t')[0] ):
# print 'new_key------------>',grid_mac
        for user,infos in itertools.groupby(data,lambda x:x.split('\t')[1]):
#           print 'new_user------------>',user
#            for info in infos:
#                print info
            block = np.genfromtxt(infos,dtype = dt,delimiter = '\t')
            user_fp_group(block,grid_mac,user)
                

    return
    data = np.genfromtxt(data, dtype = dt,delimiter = '\t')
#   dists = dist.pdist()
#    data =[ {'a':1,'b':2,'c':1},{'a':2,'b':1,'d':1}]
#    dists = dist.pdist(data,get_cos_metric)
    dists = get_pdist(data,60)
    print dists[:10]
    clusters = hcluster.linkage(dists,method = 'average')
    r = [ hcluster.fcluster(clusters,dist/10.0+0.1,'distance') for dist in range(0,10) ]
    dist = 0
    for _ in r:
        print 'cluster distance:%.1f ,cluster num:%d' % (dist,len(np.unique(_)))
        dist += 0.1
def test():
    a = '44add9ba6720;10;|c83a350f9550;9;|000000000000;13;|f41fc25705c0;8;|f41fc256bf50;9;|54e6fc5d1268;12;|e005c59c5ac4;8;|8c210aa9c4f8;9;|44add9bb77b0;10;|f41fc256da70;8;'
    b = '44add9ba6720;12;|f41fc256a410;10;|c83a350f9550;16;|000000000000;13;|f41fc256bf50;10;|54e6fc5d1268;13;|8c210aa9c4f8;11;|c83a353cd550;9;|44add9bb77b0;10;|f41fc256da70;9;'
    a = '001b2fb1c163;38|14cf92056ade;12|ea64c72d8297;9|d02db3651d1f;8|001cc2030872;8|da64c72d8297;8|c864c72d8297;8|8c210a25231e;7|da64c72d7ca7;7'
    b = '001b2fb1c163;37|14e6e4c0fec2;10|14cf92056ade;6'
    a= 'c864c7276a43;41|c864c7276a44;41|c864c7276a46;40|c864c7276a45;38|001f7aea097a;36|001f7aea097b;36|1cfa68a053d2;36|001f7a0a7c0b;22|001f7a0a7c0a;21|f8a45f9e1c6b;11'
    b= 'c864c7276a43;42|c864c7276a44;42|c864c7276a45;42|c864c7276a46;42|001f7aea097a;39|001f7aea097b;39|1cfa68a053d2;38|001f7a0a7c0a;19|001f7a0a7c0b;19'
    print get_cos_metric(str_to_wf(a),str_to_wf(b))
    print get_pearson_corr(str_to_wf(a),str_to_wf(b))

def fp_all_group(data):
    block = np.genfromtxt(get_data(data),dtype = dt,delimiter = '\t')
    user_fp_group(block,'all','all',filter = 'none',merge = False,thr=0.3)

if __name__ == '__main__'        :
#    fp_group(sys.stdin)
     fp_all_group(sys.stdin)
#    test()
        
