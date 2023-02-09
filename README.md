# Learning topology-preserving data representations (ICLR 2023)

This repository contains code base for the paper which introduces a novel method for global structure preserving dimensionality rediction based on Topological Data Analysis (TDA), specifically Representation Topology Divergence (RTD).

## Description

The proposed method aims to provide topological similarity between the data manifold and its latent representation via enforcing the similarity in topological features (clusters, loops, 2D voids, etc.) and their localization. The core of the method is the minimization of the Representation Topology Divergence (RTD) between original high-dimensional data and low-dimensional representation in latent space. RTD minimization provides closeness in topological features with strong theoretical guarantees. We develop a scheme for RTD differentiation and apply it as a loss term for the autoencoder. The proposed method “RTD-AE” better preserves the global structure and topology of the data manifold than state-of-the-art competitors as measured by linear correlation, triplet distance ranking accuracy, and Wasserstein distance between persistence barcodes.

## Dependencies

The code base requires Python 3.8.X and installation of packages listed in `requirements.txt`. 

## Usage

For reproducing experiments and training RTD-AE, please refer to `AE training.ipynb` notebook. 
For training benchmark models such as UMAP and t-SNE, please refer to `TSNE and UMAP.ipynb` notebook.
For visualization purposes, please refer to `Visualization and metrics.ipynb`.

## Cite us

Our paper is accepted to the ICLR 2023 main conference and is selected for poster session.
