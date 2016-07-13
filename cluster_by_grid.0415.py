import numpy as np
import optics
import sys
import scipy.spatial.distance as dist
import itertools 
import scipy.cluster.hierarchy as hcluster
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy.linalg as npla
import matplotlib.pyplot as plt
import math
import test_pic

def get_rd(rd,keep_num):
    high = np.max(rd)
    low = np.min(rd)
    print rd,low,high
    mid = (low + high) / 2.0
    knum = len(np.nonzero(rd < mid)[0])
    while abs(low - high) > 1e-5:
        mid = (low + high) / 2.0
        knum = len(np.nonzero(rd < mid)[0])
        if knum == keep_num:
            print 'mid2',mid,abs(low-high),abs(low-high)<1e-5,'knum:',knum,'target knum:',keep_num
            return mid
        if knum > keep_num:
            high = mid
        if knum < keep_num:
            low = mid
    print 'mid',mid,abs(low-high),abs(low-high)<1e-5,'knum:',knum,'target knum:',keep_num
    return mid
 
def get_data(file):
    for line in file:
        yield line.strip()


def get_euclid_dist(xy1,xy2):
    return float(float(xy1[0] - xy2[0]) ** 2 + float(xy1[1] - xy2[1])**2 ) ** 0.5
 
def get_cos_metric(wf1,wf2):
    if len(wf1) == 1 and len(wf2) == 1:
        k = wf1.keys()[0]
        if k in wf2 and wf1[k] > 15 and wf2[k] > 15:
            return abs(wf1[k]-wf2[k])/100.0
        return 1
            
    a = b1 = b2 = 0
    for mac in wf1.keys():
        if mac in wf2.keys():
            a += wf1[mac] * wf2[mac]

    for sig in wf1.values():
        b1 += sig**2
    for sig in wf2.values():
        b2 += sig**2
    if b1 == 0 or b2 == 0:
        return 0
    return a / ((b1** 0.5) * (b2**0.5))

class cluster_fps_by_grid:
    def __init__(self,data):
#        self.mac2normalizedmac = {}
        self.mac2index = {}
        self.data = data
        self.ap_keep_num = 28 #cut wflist
        self.dist_thr = 100 #min dist for cosine
        self.output_merge_thr = 10
        self.svd_cluster_thr = 0.6
        self.do_normalization = True
        self.merge_mac = True
        #build pdist and sparse sp matrix
        #caculate pdist
 
    def build_mac2index2(self,macs):
        ind = 0
        for mac in macs:
            if mac not in self.mac2index:
                self.mac2index[mac] = ind
                ind += 1

    def build_mac2index(self,macs):
        idx = 0
        for i in xrange(len(macs)):
            match = False
            for k in self.mac2index.keys():
                 if abs(macs[i] - k) < (1<<12): 
                     #only last 3 character changed
                     self.mac2index[macs[i]] = self.mac2index[k]
                     match = True
                     break
            if not match:
                 self.mac2index[macs[i]] = idx
                 idx += 1

    def wf2indexwf(self,wf,convert_sig = False):
        tmp = {}
        for (k,v) in wf.iteritems():
            if k == 0 or k & 0xfffffff > 0xfffffff:
                continue

            if k in self.mac2index:
                k = self.mac2index[k]
            else:
                print 'error:not in mac2index',k
                continue

            if k not in tmp:
                tmp[k] = []
            tmp[k].append(v)
        if convert_sig:
            return {k:100-min(v) for (k,v) in tmp.iteritems() }
        return {k:max(v) for (k,v) in tmp.iteritems() }
     
    def build_matrixs(self,convert_sig):
        wfs = []
        macs = set()
        data_size = self.data.shape[0]
        for i in xrange(0,data_size):
            try:
                wf = self.str_to_wf(self.data['wf_list'][i],convert_sig)
            except:
                t,v = sys.exc_info()[:2]
                print t,v,convert_sig
                #error!
                print self.str_to_wf(self.data['wf_list'][i],convert_sig)
                print self.data['wf_list'][i]
                return
            wfs.append(wf) 
            [ macs.add(k) for k in wf.keys()]

        if self.merge_mac:
            self.build_mac2index(sorted(list(macs)))
        else:
            self.build_mac2index2(sorted(list(macs)))

        #build wf matrix for svd
        ds = self.data.shape[0]
        ij = np.zeros( (2,ds * self.ap_keep_num) )
        fill = np.zeros((ds * self.ap_keep_num))

        fill_idx = 0
        for i in xrange(len(wfs)):
            wfs[i]= self.wf2indexwf(wfs[i])
            for (k,v) in wfs[i].iteritems():
               # (ij[0,fill_idx],ij[1,fill_idx],fill[fill_idx]) = (i,k,v) #row fp ,col:k mac idx
                (ij[0,fill_idx],ij[1,fill_idx],fill[fill_idx]) = (k,i,v) #row k ,col:fp
                fill_idx += 1

        self.sps_matrixs = sp.csr_matrix((fill,ij))
        self.density_matrix= self.sps_matrixs.todense()
        
        if self.do_normalization:
            self.normalization()
            max = np.max(self.macind2macnum)
#            print 'max',max
            a = 30.0 / np.log2(max)
            fill_idx = 0
            for i in xrange(len(wfs)):
    #            wfs[i]= self.wf2indexwf(wfs[i])
                for (k,v) in wfs[i].iteritems():
                    if self.macind2macnum[k] > 1:
                        v -= a * np.log2(self.macind2macnum[k])
#                        print np.log2(self.macind2macnum[k])
                    #(ij[0,fill_idx],ij[1,fill_idx],fill[fill_idx]) = (i,k,v) #row fp ,col:k mac idx
                    (ij[0,fill_idx],ij[1,fill_idx],fill[fill_idx]) = (k,i,v) #row k ,col:fp
                    fill_idx += 1
    
            self.sps_matrixs = sp.csr_matrix((fill,ij))
     
        #build dist matrix for cluster
        
        if True: #for better performance
            dm = dist.pdist(self.density_matrix,'cosine')
        else:
            dm = np.zeros( (data_size * (data_size - 1)) // 2,dtype = np.double)
            k = 0
            for i in xrange(0,data_size - 1):
                for j in xrange(i + 1, data_size):
                    dm[k] = self.get_metric(wfs[i],wfs[j],[self.data[i]['x'],self.data[i]['y']],[self.data[j]['x'],self.data[j]['y']],self.dist_thr)
                    k += 1
        dm[np.abs(dm) < 1e-11]  = 0
        self.dm = dm

    def cluster_fps2(self):
        clkg = hcluster.linkage(self.dm,method = 'average') 
        coarse_r = hcluster.fcluster(clkg,0.5,criterion = 'distance')
        self.coarse_r = coarse_r


    def cluster_by_density(self): 
        tmp_r = hcluster.fcluster(self.lkg,0.45,criterion = 'distance')
        bcount = np.bincount(tmp_r)
        point_num = np.sum(bcount[bcount > 5]) 

        RD,CD,order = optics.optics(u,4,distMethod = 'cosine')
        rd_thr = get_rd(RD,point_num)
        tmp_mark = (RD < rd_thr) 
        density_id = np.arange(len(order))*-1 - 1
        tmpid = 0
        for i in range(len(tmp_mark)):
            if i > 0 and tmp_mark[i] and tmp_mark[i-1]:
                if density_id[i-1] < 0:
                    density_id[i-1] = tmpid
                    tmpid += 1
                density_id[i] = density_id[i-1]
        self.result2 = density_id
     
    def normalization(self):
        m = self.density_matrix
        macind2macnum = np.zeros(m.shape[1])
        for i in xrange(m.shape[1]):
            x = m[ np.ravel(m[:,i])>10 ]
            sig = np.median(x,axis= 0 ).flatten()
            macind2macnum[i] = sig[sig>10].shape[1]

        self.macind2macnum = macind2macnum
        

    def cluster_fps(self):
        clkg = hcluster.linkage(self.dm,method = 'average') 
        coarse_r = hcluster.fcluster(clkg,0.3,criterion = 'distance')
        self.coarse_r = coarse_r

        bcount = np.bincount(coarse_r)
        knum = len(np.nonzero(bcount > 1)[0])

        s = self.density_matrix.shape
        if len(s) >1 and s[0] > 10 and s[1] > 10 and knum < min(s) / 2:
            (u,s,vt) = la.svds(self.sps_matrixs,k = knum)
            self.u = u
            print '============'
        else:
            
            self.result = self.coarse_r
            return (clkg,clkg)
 

#rankA = npla.matrix_rank(self.sps_matrixs)
#        if rankA < 3:
        pd = dist.pdist(u,'cosine')
        pd[np.abs(pd) < 1e-11] = 0
        lkg = hcluster.linkage(pd,method = 'average')
        self.lkg = lkg

        self.result = hcluster.fcluster(lkg,self.svd_cluster_thr,criterion = 'distance')

#        self.result = hcluster.fcluster(lkg,1)

# self.result = hcluster.fclusterdata(u,0.7,metric = 'cosine', criterion = 'distance',method = 'average')
        return (lkg,clkg)


    def str_to_wf(self,wf_list,convert_sig = True):
        wf_keep_num = self.ap_keep_num
        if convert_sig:
            return {long(mac,base=16):100 - int(sig) for (mac,sig) in [ p.split(';')[:2] for p in wf_list.split('|')[:wf_keep_num] if len(p.split(';'))>1 ] }
        return {long(mac,base=16):int(sig) for (mac,sig) in [ p.split(';')[:2] for p in wf_list.split('|')[:wf_keep_num] if len(p.split(';'))>1 ] }

    def get_metric(self,wf1,wf2,xy1,xy2,thr) :
        d = get_euclid_dist(xy1,xy2)
        cos = get_cos_metric(wf1,wf2)
        import math
        if d > thr :
            r = 1.0 + math.log((d+1)/80.0,2)/2 - cos 
        else:
            r = 1.0 -  cos
        return r

    def print_merged_data(self,grid):
        #tminfo once for each user in group
        #output group big  than 20
#        result = self.result2
        result = self.result
#        result = self.coarse_r
        for id in np.unique(result):
            if id < 0:
                continue
            infos = self.data[result == id]
            x = np.mean(infos['x'])
            y = np.mean(infos['y'])
            
            user = set()
            tm_info = []
            for info in infos:
                if info['uid'] not in user:
                    tm_info.append(info['tm_info'])
                    user.add(info['uid'])

            if len(user) < self.output_merge_thr:
                continue
            #grid x y usernum tminfo fps
            print '%s\t%d\t%d\t%d\t%s\t%s' % (\
                    grid,x,y,len(user),'&'.join(tm_info),'&'.join(np.unique(infos['wf_list'])))
        



dt_merged = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('tm_info','S30')])

def plot(D,r1,r2,lkg1,lkg2):
    test_pic.plot(D,r1,r2)
    test_pic.plot_dend(D,lkg1,None,'svd')
    test_pic.plot_dend(D,lkg2,None,'orignal')


def process_group(data,grid):
    C = cluster_fps_by_grid(data)
    C.build_matrixs(convert_sig = True)
    (lkg1,lkg2) = C.cluster_fps()
#    plot(C.sps_matrixs.todense(),C.coarse_r,C.result,lkg1,lkg2)
    C.print_merged_data(grid)


def process(data):
    max_step = 10000
    for grid,indata in itertools.groupby(get_data(data),lambda x:x.split('\t')[0] ):
        block = np.genfromtxt(indata,dtype = dt_merged,comments='None',delimiter = '\t')
        if len(block.shape) < 1 or block.shape[0] < 50:
            continue
        for i in range(block.shape[0]//max_step + 1):
            process_group(block[i*max_step:(i+1)*max_step],grid)
         
        
if __name__ == '__main__':
    process(sys.stdin)
