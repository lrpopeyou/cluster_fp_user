from common import *
import sys
import optics

from get_ssid import fetch_data
mac2index = {}
ssids = []
import re
def filter_common_ssid(ssid):
    if ssid == None:
        return ''
    if re.match(r'mobi|pccw|^china$|^chin$|^chi$|wirelessnet|^wireless$|asus|^abc$|^chen$|^zhang$|^wang$|^lin$|^sun$|^yang$|i-shanghai|samsung|^office$|tr950|Air-WASU|alcatel|totolink|lewifi|linksys|tenda|tp_link|tp|apple|htc|android|tp-link|fast|stb|mercury|itv|pc_network|dlink|d-link|none|chinaunic|chinan|netgear|netcore|i-hangzhou|cu_|VIDEOPHONE|default|backup|^360|nokia|zte_|iphone|^ac0|AirMobi|link',ssid,flags=re.IGNORECASE) == None:
        return ssid
    return ''

def get_wf(wf,topn=10):
    return sorted(str_to_wf(wf).iteritems(),key = lambda x:x[1],reverse = True)[:topn]

def gen_idx(data):
    wfs = data['wf_list']
    global mac2index,ssids
    macpair = {}
    nearest_mac = {}
    weighted_pair_mac = {}
    macs = set()
    for _ in wfs:
        try:
            wf = [mac for (mac,sig) in get_wf(_,50) ]
        except:
            print _
            sys.exit()
        for i in range(len(wf) - 1):
            for j in range(i+1,len(wf)):
#        for (m1,m2) in zip(wf[:-1],wf[1:]):
                (m1,m2) = (wf[i],wf[j])
                macs.add(m1)
                macs.add(m2)
                if m1 > m2:
                    k = '%x|%x' % (m2,m1)
                else:
                    k = '%x|%x' % (m1,m2)
                if k not in macpair:
                    macpair[k] = 0
                macpair[k] += 1
    cluster_wfs(sorted(list(macs)))
    sp = sorted(macpair.iteritems(),key  = lambda x:x[1],reverse = True)
    listk = []
    for (p,c) in sp:
        (m1,m2) = p.split('|')
        (m1,m2) = (long(m1,base=16),long(m2,base=16))
#        print m1,m2,c
        if m1 not in nearest_mac:
            nearest_mac[m1] = m2
#            print '%x' % m1,'-->', '%x' % m2
            listk.append(m1)
#will drop the last mac a->b ,no b now
#    if m2 not in nearest_mac:
#            nearest_mac[m2] = m1
#            print m2,'-->',m1
#            listk.append(m2)

    idx2mac = {}
    listk.reverse()
    fd = fetch_data()
    
    (k,c) = (long(sp[0][0].split('|')[0],base=16),0)
    while c < len(sp):

#        visted = {}
#        while k in mac2index:
#            k = nearest_mac[k]
        if k in mac2index: #looped
            while len(listk) > 0 and k in mac2index:
                k = listk.pop()
        if len(listk) == 0:
            break
        mac2index[k] = c
        idx2mac[c] = k
        ssid = fd.get_ssid(k)
#        if ssid != None:
#ssids.append(ssid.decode('gbk').encode('utf-8'))
#            ssids.append(unicode(ssid,'gbk'))
#        else:
#            ssids.append('None')
        if ssid == None:
            ssids.append('')
        else:
            try:
                ssids.append(filter_common_ssid(unicode(ssid,'gbk')))
            except:
                ssids.append('')
        c += 1
        if k not in nearest_mac:
            while len(listk) > 0 and k in mac2index:
                k = listk.pop()
 

        k = nearest_mac[k]

import scipy.cluster.hierarchy as hcluster
import scipy.spatial.distance as dist


    
import matplotlib.pyplot as plt
import scipy.sparse.linalg as lag

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
        
def plot_macs(data,method='id'):
    wfs = data['wf_list']
    img = np.zeros((len(wfs),len(mac2index) + 1) )
    row = 0
    max_i = len(mac2index)
    for _ in wfs:
        r = []
        sigs = []
        for (mac,sig) in wf2clusterwf(str_to_wf(_),convert_sig = True).iteritems():
            if mac not in mac2index:
#                print mac
                continue
            img[row,mac2index[mac] + 1]  =  sig
#    print sig
            r.append(mac2index[mac] )
            sigs.append(sig)
        if len(sigs) < 1:
            continue
#        r.sort()
#        img[row,0] = r[0]
#    top3 = 
        if method == 'id':
            img[row,0] = data['id'][row]
        else:
            img[row,0] = r[np.array(sigs).argsort()[-1]]
#        for _sr in r:
#            img[row,0] += (_sr + max_i)
#        print img[row,0]
            

        row += 1
#    (u,s,vt) = lag.svds(img[1:,:],k=5)
#    print s
#    print sorted(u[:,0])
 
#    img = img[u[:,0].argsort()][1:,:]
    img = img[img[:,0].argsort()]
    (id,img) = (img[:,0],img[:,1:])
    bcount = np.bincount(data['id'])
    knum = len(np.nonzero(bcount > 1)[0]) + 1
    density_num_thr = 5
    point_num = np.sum(bcount[bcount > 5]) 
    (u,s,vt) = lag.svds(img,k = knum) 
    print knum,u.shape
    u[np.abs(u) < 1e-2] = 0
    pd = dist.pdist(u,'cosine')
    pd[np.abs(pd) < 1e-11] = 0
    lkg = hcluster.linkage(pd,method = 'average')
    rst = hcluster.fcluster(lkg,0.7,criterion = 'distance')


    RD,CD,order = optics.optics(u,4,distMethod = 'cosine')
    thr = get_rd(RD,point_num)
    print 'get_thr:',thr
    tmp_mark = (RD<0.4) 
    density_id = np.arange(len(order))*-1 - 1
    tmpid = 0
    for i in range(len(tmp_mark)):
        if i > 0 and tmp_mark[i] and tmp_mark[i-1]:
            if density_id[i-1] < 0:
                density_id[i-1] = tmpid
                tmpid += 1
            density_id[i] = density_id[i-1]
    print 'density_id:' ,density_id
    density_id = density_id[order]
#find reasonal rd

    
#    print id
#    print img[:,1:]
#    print img[id==189,20:30][:5]
#    print ssids

    total_part = 10
    global ssids
    ssids = np.array(ssids)
    step = np.max(id) // total_part
    for i in range(total_part):
        mark = (id>=i*step) & (id<=(i+1)*step)
        fig_id = id[mark]
        fig_img = img[mark,:]
        fig_svd = u[mark,:]
        fig_svd_id = rst[mark]
        fig_dbscan_id = density_id[mark]

        print '1 fig_img',fig_img.shape,'fig_svd',fig_svd.shape
        row_filter = np.sum(fig_img,axis=1)>30
        col_filter = np.sum(fig_img,axis=0)>100

        fig_id = fig_id[row_filter]
        fig_ssid = ssids[col_filter]

        fig_img = fig_img[row_filter,:]
        fig_img = fig_img[:,col_filter]

        fig_svd = fig_svd[row_filter,:]
        fig_svd_id = fig_svd_id[row_filter]
        fig_dbscan_id = fig_dbscan_id[row_filter]

        svd_filter = np.sum(fig_svd,axis=0)>0.5
        fig_svd = fig_svd[:,svd_filter]

        fig_svd2 = fig_svd[fig_svd_id.argsort()]

        fig_img2 = fig_img[fig_svd_id.argsort()]
        fig_img3 = fig_img[fig_dbscan_id.argsort()]
        print '2 fig_img',fig_img.shape,'fig_svd',fig_svd.shape,'fig_svd2',fig_svd2.shape,'fig_img2',fig_img2.shape
        
# print fig_id.shape,fig_img.shape,np.sum(fig_img,axis=0).shape
#       fig_id = fig_id[np.sum(fig_img,axis=1)>30]

#       fig_img = fig_img[:,np.sum(fig_img,axis=0)>100] #when users num small,will filter all
#       mark2 = np.sum(fig_img,axis=1)>30
#       fig_ssid = ssids[mark2]
#       print fig_ssid.shape
#       fig_img = img[mark2,:]

        plt.clf()
        plt.subplot(3, 1, 1)
        plt.title('hierarchy',fontsize = 40)
#        plt.xlabel('figurePrints',fontsize = 30)
        plt.ylabel('APs',fontsize = 30)
        plt.imshow(fig_img.T,aspect='auto')
        tk = [ str(int(fig_id[i_])) if i_<=1 or fig_id[i_] != fig_id[i_-1]  else '' for i_ in range(len(fig_id))]
        vline1 = [ tmpi - 0.5 for tmpi in range(len(fig_id)) if tmpi > 1 and fig_id[tmpi] != fig_id[tmpi - 1] ]

        fig_svd_id = sorted(fig_svd_id)
        tk2 = [ str(int(fig_svd_id[i_])) if i_<=1 or fig_svd_id[i_] != fig_svd_id[i_-1]  else '' for i_ in range(len(fig_svd_id))]
        vline2 = [ tmpi - 0.5 for tmpi in range(len(fig_svd_id)) if tmpi > 1 and fig_svd_id[tmpi] != fig_svd_id[tmpi - 1]]
        fig_dbscan_id= sorted(fig_dbscan_id)
        tk3 = [ str(int(fig_dbscan_id[i_])) if i_<=1 or fig_dbscan_id[i_] != fig_dbscan_id[i_-1]  else '' for i_ in range(len(fig_dbscan_id))]
        vline3 = [ tmpi - 0.5 for tmpi in range(len(fig_dbscan_id)) if tmpi > 1 and fig_dbscan_id[tmpi] != fig_dbscan_id[tmpi - 1]]
        print 'tk',tk
        print 'tk2',tk2
        print 'tk3',tk3
        print vline3
    #    if method == 'id':
    #    plt.yticks(range(len(img[:0])),tk)
    #    plt.gca().set_yticklabels(tk)
    #    ax.set_yticks(range(1,len(id)))
    #    ax.set_xticklabels(tk)
        for tmpx in vline1:
            plt.axvline(tmpx,color='r')
        plt.xticks(range(len(fig_id)),tk)
        plt.yticks(range(len(fig_ssid)),fig_ssid,rotation='horizontal')

#        plt.subplot(3,1,2)
#        plt.title('svd',fontsize = 50)
#        plt.xlabel('figurePrints',fontsize = 30)
#        plt.ylabel('top k',fontsize = 30)
#        plt.imshow(fig_svd.T,aspect='auto')

#plt.subplot(3,1,3)
        plt.subplot(3,1,2)
        plt.title('svd + hierarchy',fontsize = 40)
        plt.imshow(fig_img2.T,aspect = 'auto')
        for tmpx in vline2:
            plt.axvline(tmpx,color='r')
 
        plt.yticks(range(len(fig_ssid)),fig_ssid,rotation='horizontal')
        plt.xticks(range(len(fig_svd_id)),tk2)

        plt.subplot(3,1,3)
        plt.title('svd + OPTICS',fontsize = 40)
        plt.imshow(fig_img3.T,aspect = 'auto')
        for tmpx in vline3:
            plt.axvline(tmpx,color='r')
 
        plt.yticks(range(len(fig_ssid)),fig_ssid,rotation='horizontal')
        plt.xticks(range(len(fig_svd_id)),tk3)



        fig = plt.gcf()
#fig.set_size_inches(21,37)
        fig.set_size_inches(37,21)
        ax = plt.gca()
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
#plt.tight_layout() 
        plt.savefig('test'+str(i)+'.png',dpi=100)
    
       
dt = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('t','i4')])
import sys

#dt2 = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('t','i4'),('id','i4')])

dt2 = np.dtype([('tag','S23'),('uid','S34'),('wf_list','S512'),('x','i4'),('y','i4'),('id','i4')])
def fig_clusterd():
    data = np.genfromtxt(sys.stdin,dtype=dt2,delimiter='\t')


def main():
    data = np.genfromtxt(sys.stdin,dtype=dt2,delimiter='\t')
    gen_idx(data)
    plot_macs(data,method = 'id')

if __name__ == '__main__'    :
    main()
    
    










