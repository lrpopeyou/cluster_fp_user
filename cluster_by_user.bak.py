import itertools 
from common import *
import scipy.cluster.hierarchy as hcluster
import scipy.spatial.distance as dist
import numpy as np


def get_data(file,num):
    for line in file:
        line = line.strip()
        if len(line.split('\t')) != num:
            sys.stderr.write(line+'\n')
            continue
        yield line

def user_fp_group(data,key,user = '',filter = 'mid',merge = False,wf_merge_f = 'mean',thr = 0.2,merge_thr = 0.2):
    if len(data.shape) == 0 or data.shape[0] == 1:
        key_uid = '\t'.join(key.split('_')[:2])
        print '\t'.join([key_uid,'%s' % data['wf_list'],str(data['x']),str(data['y']),str(data['tm_info'])])
        return
    dists = get_pdist(data,100,convert_sig = True)
    dists[np.abs(dists) < 1e-11] = 0
#dists[dists < 0.00001] = 0
#    print data
#    print dists
#print dists
    clusters = hcluster.linkage(dists,method = 'average')
#   print clusters
    r = hcluster.fcluster(clusters,thr,'distance')
#    print thr
#    print r
    ids = np.unique(r)
    sz = []
    for id in ids:
        sz.append(data[r==id].shape[0])
    
#mid_size = max(1.1,max(sz) / 2.0)
    mid_size = max(sz) // 2
    for id in ids:
        d = data[r==id]
        if user == '':
            out_user = '&'.join(np.unique(d['uid']))
        else:
            out_user = user
        if filter == 'mid' and d.shape[0] < mid_size:
            continue
        if merge == True:
            if wf_merge_f == 'mean':
                wf = wf_to_str(get_mean_wf(d))
            if wf_merge_f == 'sample':
                wf = get_sample_wfs(d)
            if wf_merge_f == 'uniq':
                wf = '&'.join(np.unique(d['wf_list']))
            print '\t'.join([key,out_user,wf,\
                    str(np.median(d['x'])),\
                    str(np.median(d['y'])),\
                    '|'.join(d['tm_info']) ])
            continue
        for od in d:
            key_uid = '\t'.join(od['taguid'].split('_')[:2])
            print '\t'.join([key_uid,od['wf_list'],str(od['x']),str(od['y']),str(id)])
 
#dt = np.dtype([('tag','i8'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('tm_info','S10')])
dt = np.dtype([('taguid','S66'),('wf_list','S512'),('x','i4'),('y','i4'),('tm_info','S10')])
dt_merged = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('tm_info','S30')])

max_step = 10000
def  all_fp_group(data):
    for grid_mac,data in itertools.groupby(get_data(data,6),lambda x:x.split('\t')[0] ):
        block = np.genfromtxt(data,comments = None,dtype = dt_merged,delimiter = '\t')
        if len(block.shape) < 1:
            user_fp_group(block,grid_mac,'',filter = None,merge = True,thr = 0.3)
            continue

        for i in range(block.shape[0]//max_step + 1):
            user_fp_group(block[i*max_step:(i+1)*max_step],grid_mac,'',filter = None,wf_merge_f = 'uniq',merge = False ,thr = 0.3)


def fp_group(data):
    for grid_mac,data in itertools.groupby(get_data(data,5),lambda x:x.split('\t')[0] ):
        for user,infos in itertools.groupby(data,lambda x:x.split('\t')[1]):
            block = np.genfromtxt(infos,dtype = dt,comments = None,delimiter = '\t')
            if len(block.shape) < 1:
                user_fp_group(block,grid_mac,user,filter = 'mid',merge = True,thr = 0.2)
                continue
            for i in range(block.shape[0]//max_step + 1):
                user_fp_group(block[i*max_step:(i+1)*max_step],grid_mac,user,filter = 'mid',merge = True,thr = 0.2)
                

import sys
if __name__ == '__main__':
    if len(sys.argv) <= 1 or sys.argv[1] == 'byuser':
        fp_group(sys.stdin)
    if sys.argv[1] == 'bygrid':
        all_fp_group(sys.stdin)
