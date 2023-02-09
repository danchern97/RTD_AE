#!/usr/bin/python3
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix, ConvexHull
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import connected_components, shortest_path
import sknetwork.path
from sklearn.exceptions import NotFittedError
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from itertools import chain

import gudhi as gd
import gudhi.hera as hera
import PIL

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

def get_geodesic_distance(data, n_neighbors=3, **kwargs):
    kng = kneighbors_graph(data, n_neighbors=n_neighbors, mode='distance', **kwargs)
    n_connected_components, labels = connected_components(kng)
    if n_connected_components > 1:
        kng = _fix_connected_components(
                    X=data,
                    graph=kng,
                    n_connected_components=n_connected_components,
                    component_labels=labels,
                    mode="distance",
                    **kwargs
                )

    if connected_components(kng)[0] != 1:
        raise ValueError("More than 1 connected component in the end!")
    #     return shortest_path(kng, directed=False)
    print(f"N connected: {n_connected_components}")
    return shortest_path(kng, directed=False)

class FromNumpyDataset(Dataset):
    def __init__(self, data, labels=None, geodesic=False, flatten=True, scaler=None, **kwargs):
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
        if geodesic:
            self.data_dist = get_geodesic_distance(self.data, **kwargs)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            label = self.labels[idx]
        else:
            label = 0
        if hasattr(self, 'data_dist'):
            return idx, self.data[idx], label, self.data_dist[idx]
        else:
            return idx, self.data[idx], label
        
def get_latent_representations(model, data_loader):
    labels = []
    data = []
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        for x, _, y in data_loader:
#             if x.device != model.device:
#                 x.to(model.device)
            labels.append(y.numpy())
            data.append(model(x).cpu().numpy())
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

def plot_latent_tensorboard(latent, labels):
    if latent.shape[1] < 3:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(latent[:, 0], latent[:, 1], c=labels, s=20.0, alpha=0.7, cmap='viridis')
    elif latent.shape[1] == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=labels, s=1.0, alpha=0.7, cmap='viridis')
    else:
        return None
    fig.canvas.draw()
    image = np.array(PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
    plt.close(fig)
    return image
#     return np.swapaxes(np.array(fig.canvas.renderer.buffer_rgba()), -1, 1)

def plot_latent(train_latent, train_labels, model_name, dataset_name):
    if train_latent.shape[1] > 2:
        fig = plt.figure(figsize=(12, 8))
        axes = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes = vizualize_data(train_latent, train_labels, title=f"Model: {model_name}, dataset:{dataset_name}", ax=axes)
    return fig, axes

def calculate_barcodes(distances, max_dim=1):
    skeleton = gd.RipsComplex(distance_matrix = distances)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=max_dim+1)
    barcodes = simplex_tree.persistence()
    pbarcodes = {}
    for i in range(max_dim+1):
        pbarcodes[i] = [[b[1][0], b[1][1]] for b in barcodes if b[0] == i]
    return pbarcodes

def cast_to_normal_array(barcodes):
    return np.array([[b, d] for b, d in barcodes])

def calculate_wasserstein_distance(x, z, n_runs = 5, batch_size = 2048, max_dim = 1):
    if batch_size > len(x):
        n_runs = 1
    
    results = {d:[] for d in range(max_dim+1)}
    x = x.reshape(len(x), -1)
    z = z.reshape(len(z), -1)
    for i in range(n_runs):
        ids = np.random.choice(np.arange(0, len(x)), size=min(batch_size, len(x)), replace=False)
        data = x[ids]
        distances = distance_matrix(data, data)
        distances = distances/np.percentile(distances.flatten(), 90)
        
        barcodes = {'original':calculate_barcodes(distances, max_dim=max_dim)}
        
        data = z[ids]
        distances = distance_matrix(data, data)
        distances = distances/np.percentile(distances.flatten(), 90)
        barcodes['model'] = calculate_barcodes(distances, max_dim=max_dim)
        for dim in range(max_dim+1):
            original = cast_to_normal_array(barcodes['original'][dim])
            model = cast_to_normal_array(barcodes['model'][dim])
            results[dim].append(hera.wasserstein_distance(original, model, internal_p=1))
    return results
         

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
        return 0.1*hdist.max() # upper bound