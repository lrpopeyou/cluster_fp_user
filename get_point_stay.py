from grid2 import Grid
import numpy as np
import util
import sys
import itertools
import operator
from datetime import *
from common import *

def get_data(file,sep = '\t',timecol = -1):
    for line in file:
        g = line.rstrip().split(sep)
        if timecol >= 0 and ( len(g[timecol]) != 10):
            continue
        
        yield line.rstrip().split(sep)

import scipy.cluster.hierarchy as hcluster
import scipy.spatial.distance as dist
import math
sample_range = None
#sample_range = [(12953340,4834394,15000)]
#sample_range = [(12948735,4843207,3000)]
G = Grid()

def print_merge_fp(grid_ids,imei,seg_traj,seg_id,dur):
#    print seg_traj
#    print seg_id
    for id in np.unique(seg_id):
        traj = seg_traj[seg_id == id]
        if len(traj) <= 1:
            continue
        wf = get_mean_wf(traj)
        wfinfo = wf_to_str(wf)
        #t = (traj[0]['t'] + traj[-1]['t']) / 2
        t =0
        (x,y) = (np.median(traj['x']),np.median(traj['y']))
        dt = datetime.fromtimestamp(t)
        tm_info = dt.strftime('%w:%H')
        for gid in grid_ids:
        #gridid imei wf x y timinfo 
            print '%d_%s\t%s\t%d\t%d\t%s:%d' % (gid,imei,wfinfo,x,y,tm_info,int(dur/60))

def process_stays(imei,traj):
    step = 1000
    if len(traj.shape) < 1:
        return
    step_num = traj.shape[0] / step
    for i in range(step_num + 1):
#        print traj
#        print traj.shape
        if i*step >= traj.shape[0]:
            continue
        process_stay(imei,traj[i*step:(i+1)*step])
    
def process_stay(imei,traj):
#    print imei,'------------------------>',traj.shape
    r = 20
    interval = 60*8
#    wfs = wfs[:1000]
#    traj = traj[:1000]
    if len(traj.shape) < 1 or traj.shape[0] <2:
        return
    x = traj['x']
    y = traj['y']
    in_sample = False
#print x,y
    if sample_range is not None:
        for (cx,cy,cr) in sample_range:
            crange = math.sqrt(math.pow(cx-x[0],2) + math.pow(cy-y[0],2))
            if crange < cr:
                in_sample = True
                break
    #ids = grid_util.get_grid_ids(np.median(x),np.median(y),300,3)
        if not in_sample:
            return
    
    ids = G.get_gridids_with_align(np.median(x),np.median(y))
#
#    print traj
    dm = get_pdist(traj,100,convert_sig = True)
    dm[np.abs(dm) < 1e-3] = 0
#    print dm
#    print dm.shape
#lkg = hcluster.linkage(traj[...,:2],metric = 'euclidean',method = 'average')
    #print dm
#    print dm.shape
    lkg = hcluster.linkage(dm,method = 'average')
    rst = hcluster.fcluster(lkg,0.7,criterion = 'distance') #rough dist
    rst_merge = hcluster.fcluster(lkg,0.2,criterion = 'distance') #rough dist
    seg = []
    
    
    print 'rst: ',rst
    print 'rst_merge: ',rst_merge
    for i in range(len(rst) + 1):
        if i == 0 or i == len(rst) or rst[i] != rst[i-1]:
            seg.append(i)
#
    #print rst
#    print rst_merge
    print seg
    print zip(seg[:-1],seg[1:])
    
    '''
    for (s,e) in zip(seg[:-1],seg[1:]):
        seg_traj = traj[s:e]
        seg_id = rst_merge[s:e]
        itl = seg_traj[-1]['t'] - seg_traj[0]['t']
        if itl > interval:
            print_merge_fp(ids,imei,seg_traj,seg_id,itl) 
            #print itl    
    '''

def is_keep(x,y):
    return True
    grid_ids = grid_util.get_grid_ids(x,y,300,3)
    keep_ids = [ 3174103501,3175414212]
    for id in grid_ids:
        if id in keep_ids:
            return True
    return False

def get_wf_from_bg(bginfo):
    data = bginfo.split('wf=')
    if len(data) > 1:
        end = data[1].find('&')
        if end > 0:
            return data[1][:end]
    return ''
from StringIO import StringIO
def process_line(user,group):
    count = 0
    traj = []
    for g in group:
        (x,y) = (float(g[2]),float(g[3]))
        if x < 1000:
            (x,y) = util.coordtrans("wgs84","bd09mc",float(g[2]),float(g[3]))

        if not is_keep(x,y):
            continue

        if g[-1] == 'STAY_D':
            count += 1

        if count % 2 == 1 or g[-1] == 'STAY_D':
            if g[-2] == 'bg': #depend on the format!!
                wf = get_wf_from_bg(g[-4])
                g[-4] = wf
            wf = g[-4]
            if len(wf) > 5 and wf.find(';;') < 0 :
                try:
                    str_to_wf(wf)
                    traj.append('\t'.join([str(int(x)),str(int(y)),g[1],g[-4]]))
                except:
                    pass
            if g[-1] == 'IN_TRAJ' or g[-1] == 'Original': #not match
                traj = []
                count += 1
                
        if count % 2 == 0 and g[-1] == 'STAY_D':
            if len(traj) >= 2:
                traj = np.genfromtxt(StringIO('\n'.join(traj)),dtype = np.dtype([('x','i4'),('y','i4'),('t','i4'),('wf_list','S512')]),delimiter = '\t')
                process_stays(user,traj)
                #print traj.shape[0]
            traj = []
#            traj = wfs = None



def mapper():
    for user,data in itertools.groupby(get_data(sys.stdin,timecol = 1),operator.itemgetter(0)):
        '''for each user'''
#        print '--------',user
        for day,group in itertools.groupby(data,lambda k:datetime.fromtimestamp(long(k[1])).strftime('%Y%m%d')):
#            print day
            process_line(user,group)

if __name__ == '__main__':
    mapper()
 
