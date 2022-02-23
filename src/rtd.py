#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import ripserplusplus as rpp_py
from gph.python import ripser_parallel

def get_indicies(DX, rc, dim, card):
    dgm = rc['dgms'][dim]
    pairs = rc['pairs'][dim]

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for i in range(len(pairs)):
        s1, s2 = pairs[i]
        if len(s1) == dim+1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1,:][:,l1]),[len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2,:][:,l2]),[len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(dgm[i][1] - dgm[i][0])
    
    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1,4])[perm][::-1,:].flatten())
    
    # Output indices
    indices = indices[:4*card] + [0 for _ in range(0,max(0,4*card-len(indices)))]
    return list(np.array(indices, dtype=np.int32))

def Rips(DX, dim, card, n_threads, engine):
    # Parameters: DX (distance matrix), 
    #             dim (homological dimension), 
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)
    if dim < 1:
        dim = 1
        
    if engine == 'ripser':
        DX = DX.numpy()
        DX = (DX + DX.T) / 2.0 # make it symmetrical
        DX -= np.diag(np.diag(DX))
        rc = rpp_py.run("--format distance --dim " + str(dim), DX)
    elif engine == 'giotto':
        rc = ripser_parallel(DX, maxdim=dim, metric="precomputed", collapse_edges=False, n_threads=n_threads)
    
    all_indicies = [] # for every dimension
    for d in range(1, dim+1):
        all_indicies.append(get_indicies(DX, rc, d, card))
    return all_indicies

class RTD_differentiable(nn.Module):
    def __init__(self, dim=1, card=50, mode='quantile', n_threads=25, engine='giotto'):
        super().__init__()

        if mode != 'quantile' and mode != 'median':
            raise ValueError('Only "quantile" or "median" modes are supported')
            
        self.dim = dim
        self.card = card
        self.mode = mode
        self.n_threads = n_threads
        self.engine = engine
        
    def forward(self, Dr1, Dr2):
        # inputs are distance matricies
#         d, c = self.dim, self.card
        device = Dr1.device
        
        assert device == Dr2.device, "r1 and r2 are on different devices"
        
        if Dr1.shape[0] != Dr2.shape[0]:
            raise ValueError('Point clouds must have same size')
        
        # Compute distance matrices
#         Dr1 = torch.cdist(r1, r1)
#         Dr2 = torch.cdist(r2, r2)

        Dr12 = torch.minimum(Dr1, Dr2)
        Dzz = torch.zeros((len(Dr1), len(Dr1)), device=device)

        DX = torch.cat((torch.cat((Dzz, Dr1), 1), torch.cat((Dr1, Dr12), 1)), 0)
        
        # Compute vertices associated to positive and negative simplices 
        # Don't compute gradient for this operation
        all_ids = Rips(DX.detach().cpu(), self.dim, self.card, self.n_threads, self.engine)
        all_dgms = []
        for ids in all_ids:
            # Get persistence diagram by simply picking the corresponding entries in the distance matrix
            if self.dim > 0:
                tmp_idx = np.reshape(ids, [2*self.card,2])
                dgm = torch.reshape(DX[tmp_idx[:, 0], tmp_idx[:, 1]], [self.card,2])
            else:
                tmp_idx = np.reshape(ids, [2*self.card,2])[1::2,:]
                dgm = torch.cat([torch.zeros([self.card,1], device=device), torch.reshape(DX[tmp_idx[:,0], tmp_idx[:,1]], [self.card,1])], 1)
            all_dgms.append(dgm)
        return all_dgms
    
class RTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, mode='quantile', n_threads=25, engine='giotto', is_sym=True, **kwargs):
        super().__init__()

        if mode != 'quantile' and mode != 'median':
            raise ValueError('Only "quantile" or "median" modes are supported')
        
        self.is_sym = is_sym
        self.rtd = RTD_differentiable(dim, card, mode, n_threads, engine)
    
    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd(x_dist, z_dist)
        if self.is_sym:
            rtd_zx = self.rtd(z_dist, x_dist)
        for d, rtd in enumerate(rtd_xz): # different dimensions
            loss_xz += torch.sum(rtd_xz[d][:, 1] - rtd_xz[d][:, 0])
            if self.is_sym:
                loss_zx += torch.sum(rtd_zx[d][:, 1] - rtd_zx[d][:, 0])
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss