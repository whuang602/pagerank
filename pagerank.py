import numpy as np
import numpy.linalg as npla

import scipy
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla

def pagerank2(E, return_vector = False, max_iters = 1000, tolerance = 1e-6, m = 0.15):
    """compute page rank from dense adjacency matrix
    Inputs:
    E: adjacency matrix with links going from cols to rows.
        E is a matrix of 0s and 1s, where E[i,j] = 1 means 
        that web page (vertex) j has a link to web page i.
    return_vector = False: If True, return the eigenvector as well as the ranking.
    max_iters = 1000: Maximum number of power iterations to do.
    tolerance = 1e-6: Stop when the eigenvector norm changes by less than this.
    m = 0.15: default
      
    Outputs:
      ranking: Permutation giving the ranking, most important first
      vector (only if return_vector is True): Dominant eigenvector of PageRank matrix
    This computes page rank by the following steps:
    1. Add links from any dangling vertices to all vertices.
    2. Scale the columns to sum to 1.
    3. Add a constant matrix to represent jumping at random 15% of the time.
    4. Find the dominant eigenvector with the power method.
    5. Sort the eigenvector to get the rankings.
    
    The function takes input E as a scipy csr_sparse matrix, and then never creates 
    a full matrix or any large matrix other than E.
    """
    rows, cols = E.shape
    assert(rows==cols), "E is not a square matrix"
    if(not scipy.sparse.issparse(E)):
        E = sparse.csr_matrix(E)
    nz = E.count_nonzero()
    assert np.max(E) == 1 and np.sum(E) == nz, "E is not an adjacency matrix"

    outdegree = np.array(E.sum(axis=0))
    outdegree = outdegree.flatten()

    v = np.ones(rows)
    zero_out = np.where(outdegree == 0)
    
    for iteration in range(max_iters):
        oldv = v

        # SV Calculation
        SV = np.empty(rows)
        SV.fill(np.sum(v) / rows)
        
        # EV calculation
        EV = E @ (v / outdegree)
        
        # FV calculation
        FV = np.ones(rows)*np.sum(v[zero_out])
        FV[zero_out] = FV[zero_out] - v[zero_out]
        FV = FV / (rows-1)
        
        # MV calculation
        MV = (1-m) * (EV + FV) + m*SV
        v = MV / npla.norm(MV)
        
        # End loop once accurate enough
        if (npla.norm(v - oldv) < tolerance):
            break

    v = MV
    eigval = npla.norm(v)
    v = v / eigval
    
    if npla.norm(v - oldv) < tolerance:
        print('Dominant eigenvalue is %f after %d iterations.\n' % (eigval, iteration+1))
    else:
        print('Did not converge to tolerance %e after %d iterations.\n' % (tolerance, max_iters))
    
    assert np.all(v > 0) or np.all(v < 0), 'Error: eigenvector is not all > 0 or < 0'
    vector = np.abs(v)
    ranking = np.argsort(np.abs(v))[::-1]
    if return_vector:
        return ranking, vector
    else:
        return ranking

# %%
# Test Case 1
E = np.load("PageRankEG1.npy")
r, v = pagerank2(E, return_vector = True)
print("r =", r)
print("v =", v)

# %%
# Test Case 2
E = np.load('PageRankEG3.npy')
sitename = open('PageRankEG3.nodelabels').read().splitlines()
r, v = pagerank2(E, return_vector = True)
print('r[:10] =', r[:10])
print()
for i in range(10):
    print('rank %d is page %3d: %s' % (i, r[i], sitename[r[i]]))

# %%
# Test Case 3
E = scipy.sparse.load_npz('webGoogle.npz')  
r, v = pagerank2(E, return_vector = True)
np.set_printoptions(precision = 6, suppress=True)
print("r =", r)
print("v =", v)
print()
print('Largest 10 elements of v:')
print('[ ', end='')
for i in range(10):
    print('%.4f' % v[r[i]] , end='')
    if (i != 9 ):
        print(', ',end='')
print(' ]\n')
print('Smallest 10 elements of v:')
print('[ ', end='')
for i in range(-1,-11,-1):
    print('%.4f' % v[r[i]], end='')
    if (i != -10 ):
        print(', ',end='')
print(' ]\n')
# %%
