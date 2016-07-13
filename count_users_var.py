
import itertools
import operator
import sys
import numpy as np

def get_data(file):
    for line in file:
        g =  line.strip().split()
        yield [g[0],g[1],'_'.join(g[2:-2]) ,g[-2],g[-1]]


def get_wf(wf):
    try:
        wf2 = dict({ long(mac,base=16):int(sig) for (mac,sig) in [ tp.split(';')[:2]   for tp in wf.split('|') ] if len(sig) > 1 and len(mac) > 1 and long(mac,base=16) !=0 and long(mac,base=16) & 0xffff != 0xffff })
    except:
        return None
    wf = wf2 
    if len(wf) == 0:
        return None
    return np.fromiter( [abs(wf[k]) for k in sorted(wf.keys()) ],dtype = int )

from scipy.cluster import hierarchy
import scipy.spatial.distance as sci_dist


def process(tag,infos,wf_lists,count):
    if wf_lists == None or infos == None:
        return

    x = infos['x']
    y = infos['y']
    imeis = infos['imei']
#wf_lists = np.fromiter(wf_lists,dtype = np.array)

    std_x = np.std(x)
    std_y = np.std(y)
    users_num = len(np.unique(imeis))
    if users_num < 3:
        return 
    if len(wf_lists.shape) < 2 or wf_lists.shape[1] < 2:
        return
    dists = sci_dist.pdist(wf_lists,'cosine')        
    dists[(dists < 1e-10)] = 0
    clusters = hierarchy.linkage(dists,method ='average')
    r = hierarchy.fcluster(clusters,0.3,'distance')

    for c in np.unique(r):
        idx = (r==c)
        c_x = np.median(x[idx] )
        c_y = np.median(y[idx] )
        c_std_x = np.std(x[idx])
        c_std_y = np.std(y[idx])
        c_user = len(np.unique(imeis[idx]))
        wfs = wf_lists[idx]
        wf =  np.sum(wfs,axis=0) / len(wfs)
        wf = [ '%d' % sig for sig in wf ]
        print '%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d' % (tag,'\t'.join(wf),c_x,c_y,c_user,std_x,std_y,c_std_x,c_std_y,count)
 
def main():
    for tag,data in itertools.groupby(get_data(sys.stdin), operator.itemgetter(0) ) :
        infos = None
        wf_lists = None
        l = len(tag.split('&'))
        c = 0
        for g in data:
            wf = get_wf(g[2])
            if wf == None:
                continue
            if len(wf) != l:
                continue
            if wf_lists == None:
                wf_lists = wf
            else:
                wf_lists = np.vstack((wf_lists,wf))
#            wf_lists.append(get_wf(g[2]))
            dt = np.dtype([('imei',np.str_,16),('x',np.int32),('y',np.int32)])  
#info = np.fromiter([g[1],int(g[3]),int(g[4])],dtype = dt )
            info = np.fromiter([(g[1],int(g[3]),int(g[4]))],dtype = dt )
            if infos == None:
                infos = info
            else:
                infos = np.vstack((infos,info))

            if len(wf_lists) > 15000:
                process(tag,infos,wf_lists,c)
                c += 1
                infos = None
                wf_lists = None
        process(tag,infos,wf_lists,c)
   
if __name__ == '__main__'            :
    main()
