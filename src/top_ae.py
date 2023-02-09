"""Topolologically regularized autoencoder using approximation."""
import numpy as np
import torch
import torch.nn as nn

from .topology import PersistentHomologyCalculation #AlephPersistenHomologyCalculation, \
import pytorch_lightning as pl

from .utils import plot_latent_tensorboard, calculate_wasserstein_distance

class TopologicallyRegularizedAutoencoder(pl.LightningModule):
    """Topologically regularized autoencoder."""

    def __init__(self, encoder, decoder, MSELoss=None, rtd_l=1., toposig_kwargs=None, **kwargs):
        """Topologically Regularized Autoencoder.

        Args:
            lam: Regularization strength
            ae_kwargs: Kewords to pass to `ConvolutionalAutoencoder` class
            toposig_kwargs: Keywords to pass to `TopologicalSignature` class
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rtd_l = rtd_l
        self.MSELoss = MSELoss
        toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True)
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, x_distances, y = train_batch
        latent = self.encoder(x)
        x_distances = x_distances / x_distances.max()
        latent_distances = torch.cdist(latent, latent, p=2)
        latent_distances = latent_distances / self.latent_norm
        x_reconstructed = self.decoder(latent)
        if self.MSELoss is not None:
            mse_loss = self.MSELoss(x_reconstructed, x)
            self.log('train/mse_loss', mse_loss)
        else:
            mse_loss = 0.0
        topo_loss, topo_loss_components = self.topo_sig(x_distances, latent_distances)
        topo_loss = topo_loss / float(x.shape[0]) 
        self.log('train/top_loss',  topo_loss)
        loss = mse_loss + self.rtd_l * topo_loss
        self.log('train/loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_distances, y = val_batch
        latent = self.encoder(x)
        x_distances = x_distances / x_distances.max()
        latent_distances = torch.cdist(latent, latent, p=2)
        latent_distances = latent_distances / self.latent_norm
        x_reconstructed = self.decoder(latent)
        if self.MSELoss is not None:
            mse_loss = self.MSELoss(x_reconstructed, x)
            self.log('val/mse_loss', mse_loss)
        else:
            mse_loss = 0.0
        topo_loss, topo_loss_components = self.topo_sig(x_distances, latent_distances)
        topo_loss = topo_loss / float(x.shape[0]) 
        self.log('val/top_loss',  topo_loss)
        loss = mse_loss + self.rtd_l * topo_loss
        self.log('val/loss', loss)
        return x, latent, y
    
    def validation_epoch_end(self, validation_step_outputs):
        logger = self.logger.experiment
        if self.current_epoch % 5 == 0:
            xs, zs, ys = [], [], []
            for x, z, y in validation_step_outputs:
                xs.append(x.cpu().detach().numpy())
                zs.append(z.cpu().detach().numpy())
                ys.append(y.cpu().detach().numpy())
            x = np.concatenate(xs, axis=0)
            z = np.concatenate(zs, axis=0)
            y = np.concatenate(ys, axis=0)
            image = plot_latent_tensorboard(z, y)
            if image is not None:
                logger.add_image('val/image', image, self.current_epoch, dataformats='HWC')
        if self.current_epoch % 20 == 0:
            wass = calculate_wasserstein_distance(x, z, batch_size=2048, max_dim=0)
            logger.add_scalar('val/wasserstein_h0', np.mean(wass.get(0, 0.0)), self.current_epoch)
            logger.add_scalar('val/wasserstein_h1', np.mean(wass.get(1, 0.0)), self.current_epoch)


class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles

        self.match_edges = match_edges

        # if use_cycles:
        #     use_aleph = True
        # else:
        #     if not sort_selected and match_edges is None:
        #         use_aleph = True
        #     else:
        #         use_aleph = False

        # if use_aleph:
        #     print('Using aleph to compute signatures')
        ##self.signature_calculator = AlephPersistenHomologyCalculation(
        ##    compute_cycles=use_cycles, sort_selected=sort_selected)
        # else:
        print('Using python to compute signatures')
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components
