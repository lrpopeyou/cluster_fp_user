import numpy as np
from scipy.cluster import hierarchy
import scipy.spatial.distance as sci_dist

data = np.random.rand(15000,10)
dists = sci_dist.pdist(data,'cosine')
dists[(dists < 1e-10)] = 0
clusters = hierarchy.linkage(dists,method ='average')
r = hierarchy.fcluster(clusters,0.3,'distance')
print len(r)

import sys
for line in sys.stdin:
    print line
