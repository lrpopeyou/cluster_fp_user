import util
from grid import grid_util

def get_aplist(wf_list):
    min_sig = 101
    orig_wf_list = [ wf.split(';')[:2] for wf in wf_list.split('|') if len(wf) > 1 ]
    wf_list = sorted(orig_wf_list,key = lambda x:(abs(int(x[1]))<<16 | (long(x[0],base=16) &0xff)))
#    if len(orig_wf_list) > 3:
#        min_sig = 95
#    if len(orig_wf_list) > 5:
    top_n = 100
    #wf_list = [ long(a,base=16) for (a,b) in wf_list[:top_n] if len(b)>1 and abs(int(b)) < min_sig ]
#    wf_list = [ long(wf.split(';')[0],base=16) for wf in wf_list.split('|') if len(wf) > 1 ]
    #return '&'.join([ '%x' % mac for mac in sorted(wf_list) if mac != 0 and mac & 0xffff != 0xffff ])

    #return '&'.join([ '%x' % mac for mac in sorted(wf_list) if mac != 0 and mac & 0xffff != 0xffff ])
    sig2w_wf_list = [(mac,100-abs(int(sig))) for (mac,sig) in wf_list[:top_n] ]
    return '|'.join([ '%s;%d' % (mac,sig) for (mac,sig) in sig2w_wf_list ]) 


from collections import Counter 

userposition = Counter()

def is_uniq(im,wf):    
    _k = '%s@%s' % (im,wf)
    if _k in userposition:
        return False
    userposition[_k] += 1
    return True
def process(line):
    g = line.strip().split('\t')
    im = g[0]
    wf = g[-1]
    if len(wf) < 6: #1 and more ap
        return
    wf_list = get_aplist(wf)
    x = float(g[1])
    y = float(g[2])
    t = int(g[3])
    if x < 10000:
        (x,y) = util.coordtrans('wgs84ll','bd09mc',x,y)
    if  not is_uniq(im,wf_list):
        return
    grid_ids = grid_util.get_grid_ids(x,y,300,3)
    for id in grid_ids:
        print '%d\t%s\t%s\t%d\t%d\t%d' % (id,im,wf_list,x,y,t)

import sys
if __name__ == '__main__'    :
    for line in sys.stdin:
        try:
            process(line)
        except :
#            exctype, value = sys.exc_info()[:2]
#            sys.stderr.write('%s\t\%s\t%s\n' %  (exctype,value,line ))
            pass

