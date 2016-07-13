import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import numpy as np
def plot_dend(D,lkg,u,name):
    fig = plt.figure(figsize=(16,9))
#  ax1 = fig.add_axes([0.09,0.1,0.2,0.6])

#mpute and plot second dendrogram.
    ax2 = fig.add_axes([0.09,0.71,0.8,0.2])
    Z2 = hcluster.dendrogram(lkg)
#    ax2.set_xticks([])
#    ax2.set_yticks([])

#ot distance matrix.
    axmatrix = fig.add_axes([0.09,0.1,0.8,0.6])
    idx2 = Z2['leaves']
    D = D[idx2,:]
#    D = D[:,np.sum(D,axis = 0) > 30]
    im = axmatrix.imshow(D.T, aspect='auto', origin='lower')
#    axmatrix.set_xticks([])
#    axmatrix.set_yticks([])

#ot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    fig.savefig(name)


def plot(orig,id1,id2):
#    print orig.shape,id1.shape,id2.shape
#    print orig
#    print id1
#    print id2
    total_part = 1
    step = np.max(id1) // total_part
    img = orig[id1.argsort()]
    id2 = id2[id1.argsort()]
    id1 = np.sort(id1)
    print orig.shape,id1.shape,id2.shape
#  step = 10
    for i in range(total_part):
        mark = (id1>=i*step) & (id1<=(i+1)*step)
        fig_img = orig[mark,:]
        fig_id = id1[mark]
        fig_svd_id = id2[mark]

        print '1 fig_img',fig_img.shape,'fig_id',fig_id.shape,'fig_svd_id',fig_svd_id.shape
        row_sum = np.sum(fig_img,axis=1)
        col_sum = np.sum(fig_img,axis=0)
#        c = (row_sum > 30) * (col_sum > 10)
        
        row_filter = (np.sum(fig_img,axis=1) > 30)
        col_filter = (np.sum(fig_img,axis=0)>100)

#        print fig_id.shape,fig_id
        row_filter = np.ravel(row_filter)
        col_filter = np.ravel(col_filter)
#       print row_filter.shape,row_filter
        if False:
            fig_id = fig_id[row_filter]
            fig_img = fig_img[row_filter,:]
#       print 'aaaa',fig_img.shape
            fig_img = fig_img[:,col_filter]
#       print col_filter
#       print 'aaaa',fig_img.shape
            fig_svd_id = fig_svd_id[row_filter]
#fig_img = fig_img[c]
#       print col_filter
#       print fig_img.shape

#        print row_filter
#        print fig_img.shape,fig_svd_id.shape
#       print 'fs',fig_img.shape
        fig_img2 = fig_img[fig_svd_id.argsort(),:]
        fig_img = fig_img[fig_id.argsort()]
        print '2 fig_img',fig_img.shape,'fig_img2',fig_img2.shape
        
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.title('hierarchy',fontsize = 40)
        plt.ylabel('APs',fontsize = 30)
        plt.imshow(fig_img.T,aspect='auto')
        tk = [ str(int(fig_id[i_])) if i_<=1 or fig_id[i_] != fig_id[i_-1]  else '' for i_ in range(len(fig_id))]
        vline1 = [ tmpi - 0.5 for tmpi in range(len(fig_id)) if tmpi > 1 and fig_id[tmpi] != fig_id[tmpi - 1] ]

        fig_svd_id = sorted(fig_svd_id)
        tk2 = [ str(int(fig_svd_id[i_])) if i_<=1 or fig_svd_id[i_] != fig_svd_id[i_-1]  else '' for i_ in range(len(fig_svd_id))]
        vline2 = [ tmpi - 0.5 for tmpi in range(len(fig_svd_id)) if tmpi > 1 and fig_svd_id[tmpi] != fig_svd_id[tmpi - 1]]
        
        for tmpx in vline1:
            plt.axvline(tmpx,color='r')
        plt.xticks(range(len(fig_id)),tk)
#        plt.yticks(range(len(fig_ssid)),fig_ssid,rotation='horizontal')

#plt.subplot(3,1,3)
        plt.subplot(2,1,2)
        plt.title('svd + hierarchy',fontsize = 40)
        plt.imshow(fig_img2.T,aspect = 'auto')
        for tmpx in vline2:
            plt.axvline(tmpx,color='r')
 
#        plt.yticks(range(len(fig_ssid)),fig_ssid,rotation='horizontal')
        plt.xticks(range(len(fig_svd_id)),tk2)

#        plt.subplot(3,1,3)
#        plt.title('svd + OPTICS',fontsize = 40)
#        plt.imshow(fig_img3.T,aspect = 'auto')
#        for tmpx in vline3:
#            plt.axvline(tmpx,color='r')
 
#        plt.yticks(range(len(fig_ssid)),fig_ssid,rotation='horizontal')
#        plt.xticks(range(len(fig_svd_id)),tk3)


        fig = plt.gcf()
#fig.set_size_inches(21,37)
        fig.set_size_inches(37,21)
        ax = plt.gca()
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
#plt.tight_layout() 
        plt.savefig('test_f'+str(i)+'.png',dpi=100)
 

