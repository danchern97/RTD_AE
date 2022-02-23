#!/usr/bin/python3
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix, ConvexHull
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.exceptions import NotFittedError
from scipy.spatial.distance import cdist
from itertools import chain

def get_linear_model(input_dim, latent_dim=2, n_hidden_layers=2, hidden_dim=32, m_type='encoder', **kwargs):
    layers = list(
        chain.from_iterable(
            [
                (nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_hidden_layers)
            ]
        )
    )
    if m_type == 'encoder':
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()] + layers + [nn.Linear(hidden_dim, latent_dim)]
    elif m_type == 'decoder':
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()] + layers + [nn.Linear(hidden_dim, input_dim)]
    return nn.Sequential(*layers)

def get_cnn_model(input_dim=(64, 64), latent_dim=2, n_hidden_layers=2, hidden_dim=32, m_type='encoder', **kwargs):
    modules = []
    width, heigth = input_dim
    if m_type == 'encoder':
        in_channels = 1
        for i in range(n_hidden_layers):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dim
        modules.append(nn.Flatten(start_dim=1, end_dim=- 1))
        modules.append(nn.Linear(int(hidden_dim*width*heigth/(4**n_hidden_layers)), latent_dim))
    elif m_type == 'decoder':
        shape = int(hidden_dim*width*heigth/(4**n_hidden_layers))
        modules.append(nn.Linear(latent_dim, shape))
        modules.append(Reshape(hidden_dim, int(width/(2**n_hidden_layers)), int(heigth/(2**n_hidden_layers))))
        for i in range(n_hidden_layers-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim, hidden_dim,
                              kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU())
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, 1,
                          kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(1),
                nn.LeakyReLU())
        )
    return nn.Sequential(*modules)
            

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view((batch_size, *self.shape))

class FromNumpyDataset(Dataset):
    def __init__(self, data, labels=None, flatten=True, scaler=None):
        if labels is not None:
            assert len(labels) == len(data), "The length of labels and data are not equal"
            self.labels = labels
        if flatten:
            self.data = torch.tensor(data).flatten(start_dim=1).numpy()
        else:
            self.data = data
        if scaler is not None:
            try:
                self.data = scaler.transform(self.data)
            except NotFittedError:
                self.data = scaler.fit_transform(self.data)
        self.scaler = scaler
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            return idx, self.data[idx], self.labels[idx]
        else:
            return idx, self.data[idx], 0
        
def get_latent_representations(model, data_loader):
    labels = []
    data = []
    model.eval()
    with torch.no_grad():
        for x, _, y in data_loader:
            labels.append(y.numpy())
            data.append(model(x).numpy())
    return np.concatenate(data, axis=0), np.concatenate(labels, axis=0)

def vizualize_data(data, labels=None, alpha=1.0, s=1.0, title="", ax=None):
    assert labels.shape[0] == data.shape[0], "Length of labels and data are not equal"
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    if data.shape[1] == 2:
        x, y = zip(*data)
        ax.scatter(x, y, alpha=alpha, c=labels, s=s)
    else:
        x, y, z = zip(*data)
        ax.scatter(x, y, z, alpha=alpha, c=labels, s=s)
    ax.set_title(title, fontsize=20)
    return ax

def plot_latent(train_latent, train_labels, model_name, dataset_name):
    if train_latent.shape[1] > 2:
        fig = plt.figure(figsize=(12, 8))
        axes = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes = vizualize_data(train_latent, train_labels, title=f"Model: {model_name}, dataset:{dataset_name}", ax=axes)
    return fig, axes

def _fix_connected_components(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    """Add connections to sparse graph to connect unconnected components.
    For each pair of unconnected components, compute all pairwise distances
    from one component to the other, and add a connection on the closest pair
    of samples. This is a hacky way to get a graph with a single connected
    component, which is necessary for example to compute a shortest path
    between all pairs of samples in the graph.
    Parameters
    ----------
    X : array of shape (n_samples, n_features) or (n_samples, n_samples)
        Features to compute the pairwise distances. If `metric =
        "precomputed"`, X is the matrix of pairwise distances.
    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples.
    n_connected_components : int
        Number of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.
    component_labels : array of shape (n_samples)
        Labels of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.
    mode : {'connectivity', 'distance'}, default='distance'
        Type of graph matrix: 'connectivity' corresponds to the connectivity
        matrix with ones and zeros, and 'distance' corresponds to the distances
        between neighbors according to the given metric.
    metric : str
        Metric used in `sklearn.metrics.pairwise.pairwise_distances`.
    kwargs : kwargs
        Keyword arguments passed to
        `sklearn.metrics.pairwise.pairwise_distances`.
    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples, with a single connected component.
    """
    if metric == "precomputed" and sparse.issparse(X):
        raise RuntimeError(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(component_labels == j)
            Xj = X[idx_j]

            if metric == "precomputed":
                D = X[np.ix_(idx_i, idx_j)]
            else:
                D = pairwise_distances(Xi, Xj, metric=metric, **kwargs)

            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)
            if mode == "connectivity":
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj]
            else:
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    return graph

class FurthestScaler:
    def __init__(self, p=2): # approximate
        self.is_fitted = False
        self.p = p
        
    def fit(self, data):
        self.furthest = self._furthest_distance(data)
        self.is_fitted = True
        
    def transform(self, data):
        if not self.is_fitted:
            raise NotFittedError
        return data / self.furthest
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def _furthest_distance(self, points, sample_frac=0.0):
        # exact solution, very computationaly expesive
        # hull = ConvexHull(points) 
        # hullpoints = points[hull.vertices,:]
        # hdist = distance_matrix(hullpoints, hullpoints, p=self.p)
        # approximation: upper bound
        # pick random point and compute distances to all of the points
        # diameter min: max(distances), diameter max (triangle inequality): 2 max(distances)
        if len(points.shape) > 2:
            points = points.reshape(points.shape[0],-1)
        idx = np.random.choice(np.arange(len(points)), size=1)
        hdist = distance_matrix(points[idx], points, p=self.p)
        return hdist.max() # upper bound