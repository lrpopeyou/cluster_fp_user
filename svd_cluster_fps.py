import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import common

def get_hash(mac,m = 123321):
    return hash(mac) % m

macs_dict = {}
max_idx = 0

def get_mac_idx(mac):
    global max_idx,macs_dict
    if mac in macs_dict:
        return macs_dict[mac]
    else:
        max_idx += 1 
        macs_dict[mac] = max_idx
        return max_idx 

dt = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S162'),('x','i4'),('y','i4'),('t','i4')])

def get_svd(data):
    input = np.genfromtxt(data,dtype = dt,delimiter = '\t')
    row = 0
    data_i = 0    
    ij = np.zeros((2,input.shape[0]*10),dtype=np.uint64)
    data = np.zeros((input.shape[0]*10),dtype=np.float)
    macs = set()
    for wf_list in input['wf_list']:
        wf = common.str_to_wf(wf_list)
        for (k,v) in wf.iteritems():
            ki = long(k,base=16)
            mask = 0xffffffffffff
            if ki == 0 or ki & mask == mask:
                continue
#k = get_hash(k)
            k = get_mac_idx(ki)
            (ij[0,data_i],ij[1,data_i],data[data_i]) = (row,k,v)
            data_i += 1
            macs.add(k)
        row += 1
   
#    print '%x,%x' % (ij[1,...].min(),ij[1,...].max())
    m = sp.csr_matrix((data,ij))        
#    print len(macs)
    (u,s,vt) = la.svds(m,k=10)
    print '\n'.join([ '\t'.join(p) for p in filter_small(u) ])

def filter_small(u,thr = 0.01):
    r = []
    (I,J) = u.shape
    for i in range(I):
        p = []
        for j in range(J):
            if u[i,j] >= thr:
                p.append('%d:%0.2f' % (j,u[i,j]))
        r.append(p)
    return r

def svd_wifis(wf_lists,hash_num,nk):
    data_size = wf_lists.shape[0]
    ij = np.zeros((2,data_size * 10))
    data = np.zeros((data_size * 10))
    row = data_i = 0
    macs = set()
    for wf_list in wf_lists:
        wf = common.str_to_wf(wf_list)
        for (k,v) in wf.iteritems():
            ki = long(k,base=16)
            mask = 0xffffffffffff
            if ki == 0 or ki & mask == mask:
                continue
#k = get_hash(k)
            k = get_mac_idx(ki)
            (ij[0,data_i],ij[1,data_i],data[data_i]) = (row,k,v)
            data_i += 1
            macs.add(k)
        row += 1
   
    m = sp.csr_matrix((data,ij))        
    (u,s,vt) = la.svds(m,k = min(nk,min(m.shape)//2))

    print m.todense()
    return u,s,vt

    
def get_data(file):
    for line in file:
        yield line.strip()

import itertools 
import sys    
def svd_users(data):
    for grid_mac,data in itertools.groupby(get_data(data),lambda x:x.split('\t')[0] ):
# print 'new_key------------>',grid_mac
        for user,infos in itertools.groupby(data,lambda x:x.split('\t')[1]):
            one_user = np.genfromtxt(infos,dtype = dt,delimiter = '\t')
            if len(one_user['wf_list'].shape) < 1:
                continue
            (u,s,vt) = svd_wifis(one_user['wf_list'],1000,5)
            fs = filter_small(u,0.01) 
            i = 0
            print '===========>',s
            for info in one_user:
                print '%s\t%s' % (info,'\t'.join(fs[i]))
                i += 1
#
if __name__ == '__main__'    :
    svd_users(sys.stdin)
#    get_svd(sys.stdin)

