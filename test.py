import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import common

r1 = [-1.03,   0.74,    -0.02 ,  0.51,    -1.31 ,  0.99,    0.69,    -0.12  , -0.72,   1.11]
r2 = [-2.23  , 1.61  ,  -0.02 ,  0.88  ,  -2.39,   2.02 ,   1.62 ,   -0.35 ,  -1.67   ,2.46]

ij = np.zeros((2,20),dtype=np.uint64)
data = np.zeros((20),dtype=np.float)
i = 0
for r in range(len(r1)):
    (ij[0,i],ij[1,i],data[i]) = (r,0,r1[r])
    i+= 1
for r in range(len(r2)):
    (ij[0,i],ij[1,i],data[i]) = (r,1,r2[r])
    i+= 1
print ij
m = sp.csr_matrix((data,ij))    
(u,s,vt) = la.svds(m,1)
print m.todense()
print u
print s
print vt
