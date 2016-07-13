import common
import numpy as np
fp2id = {}
id2count = {}
id2usernum = {}
def load_data(filename):
    global fp2id
    global id2count
    f = open(filename)
    id = 0
    for line in f:
        g = line.split('\t')
        if len(g) > 3 and int(g[3]) < 3:
            continue
        try:
            id2usernum[id]= int(g[3])
        except:
            continue
        for wf in g[5].split('&'):
            fp2id[wf] = id
            if id not in id2count:
                id2count[id] = 0
            id2count[id] += 1

        id += 1

    f.close()

def merge_similar_id(line):
    wf = common.str_to_wf(line,convert_sig = True)
    r = {}
    for (k,v) in fp2id.iteritems():
        dist =  common.get_cos_metric(wf,common.str_to_wf(k,convert_sig = True))
       # if dist >= 0.5:
        if dist > 0:
            if v not in r:
                r[v] = []
            r[v].append(dist)
    
    r2 = {}
    r3 = {}
    if len(r) == 0:
        print 'no match'
    for (id,v_list) in r.iteritems():
        v_list = np.array(v_list)
        if np.max(v_list) < 0.6 or id2usernum[id] < 20:
            continue
        print id,'-->', '%d,%d,%0.2f\t%0.2f,%0.2f,%0.2f' % (id2usernum[id],len(v_list),len(v_list)/id2count[id],np.mean(v_list),np.std(v_list),np.max(v_list))
        r3['%d-%d:%0.2f'%(id,id2usernum[id],np.max(v_list))] = np.max(v_list)
        r2[id] = np.max(v_list)
        

    if len(r2) > 0:
#print 'r2:', sorted(r2.iteritems(),key = lambda x:x[1],reverse = True)
#       print 'r3:',sorted(r3.iteritems(),key = lambda x:x[1],reverse = True)
        return sorted(r2.iteritems(),key = lambda x:x[1],reverse = True)[0][0],sorted(r3.iteritems(),key = lambda x:x[1],reverse = True)[0][0]
    else:
        return 'None','None'

import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def main():   
    load_data(sys.argv[1])
    xs = []  
    xouts = []  
    ys = []
    youts = []
    ythis = None
    xthis = None
    label = None
    labels1 = []
    labels2 = []
    i = -1
    ids = set()
    for line in sys.stdin:
        line = line.strip()
        if line.find(';') > 0:
            r,r2 = merge_similar_id(line)
            if r != 'None':
                i+= 1
                print i,r,label
                xthis[-1].append(i)    
                ythis[-1].append(r)
                ids.add(r)
    #        print '%s\t%s' % (r2,line)
        else:
            if len(line) > 2 and label != line:
                label = line
                labels2.append('out of '+ label[0])
                youts.append([])
                xouts.append([])
                ythis = youts
                xthis = xouts

            elif label != line:
                label = line
                labels1.append(label[0])
                ys.append([])
                xs.append([])
                ythis = ys
                xthis = xs
                
            print line

    ids = sorted(list(ids))
    ids = {k:v for (k,v) in zip(ids,range(len(ids))  ) }
    print ids

    rand = np.random.rand(1000) * 200
    colors = cm.rainbow(np.linspace(0, 1, max(len(ys),len(youts))))
#    colors = [ plt.get_cmap('jet')(x) for x in np.linspace(0, 0,9, max(len(ys),len(youts)))]
    print xs
    print ys
    print labels1
    print xouts
    print youts
    print labels2
    fig = plt.figure()
    ax = plt.subplot(111)
    for x,y,c,l in zip(xs,ys,colors,labels1):
        print 'before y',y
        y = [ ids[i] for i in y ]
        print 'after y',y
        plt.scatter(x,y,color=c,marker='o',label = l,facecolors='none')
#        plt.scatter(x,y,marker='o')
    for x,y,c,l in zip(xouts,youts,colors,labels2):
        y = [ ids[i] for i in y ]
        plt.scatter(x,y,color=c,marker='x',label = l)
#        plt.scatter(x,y,marker='x')
    plt.gcf().set_size_inches(12,8)
# Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
#    iplt.legend(loc = 'lower right')
    plt.savefig(sys.argv[1]+'.png')

    
if __name__ == '__main__':
    main()
