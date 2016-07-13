import numpy as np
def hash_mac(mac,size = 123721):
    return hash(mac) % size


def get_pearson_corr(wf1,wf2):
    if len(wf1) == 1:
        k = wf1.keys()[0]
        if wf1[k] > 30 and k in wf2 and wf2[k] > 30:
            return 1
        else:
            return get_cos_metric(wf1,wf2)
    elif len(wf2) == 1:
        k = wf2.keys()[0]
        if wf2[k] > 30 and k in wf1 and wf1[k] > 30:
            return 1
        else:
            return get_cos_metric(wf1,wf2)
 
    v1ba = np.average( wf1.values())
    v2ba = np.average( wf2.values())

    a = b1 = b2 = 0
    for mac in wf1.keys():
        if mac in wf2.keys():
            a += (wf1[mac] -v1ba) * (wf2[mac]-v2ba)

    for sig in wf1.values():
        b1 += (sig-v1ba)**2
    for sig in wf2.values():
        b2 += (sig-v2ba)**2
    if b1 == 0 or b2 == 0:
        return 0
    return a / ((b1** 0.5) * (b2**0.5))

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
def get_euclid_dist(xy1,xy2):
    return float(float(xy1[0] - xy2[0]) ** 2 + float(xy1[1] - xy2[1])**2 ) ** 0.5
    
def judge_by_rule(wf1,wf2):
    wf1 = sorted(wf1.iteritems(),lambda x:x[1])
    wf2 = sorted(wf2.iteritems(),lambda x:x[1])
#    if wf1[0][0] == wf2[0][0] and 

def get_metric(wf1,wf2,xy1,xy2,thr) :
    d = get_euclid_dist(xy1,xy2)
    cos = get_cos_metric(wf1,wf2)
#    pearson = get_pearson_corr(wf1,wf2)
    import math
    if d > thr :
        r = 1.0 + math.log((d+1)/80.0,2)/2 - cos 
#        print 'r1',r
    else:
#        print cos,pearson
        r = 1.0 -  cos
#        print 'r2',r,cos
    return r

def str_to_wf(wf_list,convert_sig = False) :
    if convert_sig:
        return {long(mac,base=16):100 - int(sig) for (mac,sig) in [ p.split(';')[:2] for p in wf_list.split('|')[:28] if len(p.split(';'))>1 ] }
    return {long(mac,base=16):int(sig) for (mac,sig) in [ p.split(';')[:2] for p in wf_list.split('|')[:28] if len(p.split(';'))>1 ] }

def wf_to_str(wf,convert_sig = False):
    if convert_sig:
        return '|'.join([ '%x;%d' % (mac,sig) for mac,sig in sorted(wf.iteritems(),key = lambda x:x[1],reverse = True)[:30] ])
    return '|'.join([ '%x;%d' % (mac,sig) for mac,sig in sorted(wf.iteritems(),key = lambda x:x[1])[:30] ])

wf2cluster = {}
wfkey = {}

#conver group'wifi to one
def cluster_wfs(wfs):
    global wf2cluster,wfkey
    for i in range(len(wfs)):
        match = False
        for k in wfkey.keys():
            if abs(wfs[i] - k) < (1<<12):
                wf2cluster[wfs[i]] = k
                match = True
                break
        if not match:
            wf2cluster[wfs[i]] = wfkey[wfs[i]] = wfs[i]

def wf2clusterwf(wf,convert_sig = False):
    tmp = {}
    for (k,v) in wf.iteritems():
        if k in wf2cluster:
            oldk = k
            k = wf2cluster[k]
        if k not in tmp:
            tmp[k] = []
        tmp[k].append(v)
#    print 'convert_sig:',convert_sig
    if convert_sig:
        return {k:100-min(v) for (k,v) in tmp.iteritems() }
    return {k:max(v) for (k,v) in tmp.iteritems() }
        
#get dist with tuned metric,with xy,wf
#convert sig: sig -> 100 - sig for weighted caculation
#thr: xy dist for similar 
def get_pdist(data,thr = 100,convert_sig = False): 
    wfs = []
    macs = set()
    data_size = data.shape[0]
    for i in xrange(0,data_size):
        try:
            wf = str_to_wf(data['wf_list'][i],convert_sig)
        except:
            print data['wf_list'][i]
            return
        wfs.append(wf) 
        [ macs.add(k) for k in wf.keys()]
    cluster_wfs(sorted(list(macs)))
    wfs2c = []
    for wf in wfs:
        wfs2c.append(wf2clusterwf(wf))
    wfs = wfs2c
    dm = np.zeros( (data_size * (data_size - 1)) // 2,dtype = np.double)
    k = 0
    for i in xrange(0,data_size - 1):
        for j in xrange(i + 1, data_size):
#print wfs[i]
            dm[k] = get_metric(wfs[i],wfs[j],[data[i]['x'],data[i]['y']],[data[j]['x'],data[j]['y']],thr)
            k += 1
    
    return dm

def get_median_wf(data):
    m = data.shape[0]
    wfs = {}
    for i in xrange(0,m):
        for (k,v) in str_to_wf(data[i]['wf_list']).iteritems():
            if k not in wfs:
                wfs[k] = []
            wfs[k].append(v)

    return {mac:np.median(np.array(value_list)) for mac,value_list in wfs.iteritems()}

def get_sample_wfs(data):
    return ''
def get_mean_wf(data):
    m = data.shape[0]
    wfs = {}
    for i in xrange(0,m):
        for (k,v) in str_to_wf(data[i]['wf_list']).iteritems():
            if k not in wfs:
                wfs[k] = []
            wfs[k].append(v)

    return {mac:np.average(np.array(value_list)) for mac,value_list in wfs.iteritems()}



import sys
def get_wf_from_stdin2():
    k_macs = {}
    f = open('data/macs.users','r')
    for line in f:
        g = line.rstrip().strip().split()
        if int(g[0]) > 100:
            k_macs[g[1]] = 1
    f.close()
    for line in sys.stdin:
        g = line.split('\t')
        wf = str_to_wf(g[2])
        is_k = False
        for (k,v) in wf.iteritems():
            if k in k_macs and v > 15:
               is_k = True
               break
#            print k
        if is_k:
#            print line.strip()
            print sorted(wf.keys())
def get_wf_from_stdin():
  for line in sys.stdin:
        g = line.split('\t')
        wf = str_to_wf(g[2])
        for k in wf.keys():
            print '%s\t%x' %( g[1],k)

def get_mac_hash():
    for line in sys.stdin:
        print hash_mac(line.strip())
if __name__ == '__main__'            :
    get_wf_from_stdin()
#    get_mac_hash()
