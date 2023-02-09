#!/usr/bin/python3
import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .utils import plot_latent_tensorboard, calculate_wasserstein_distance

class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, RTDLoss=None, MSELoss=None, rtd_l=0.1, rtd_every_n_batches=1, rtd_start_epoch=0, lr=5e-4, **kwargs):
        """
        RTDLoss - function of topological (RTD) loss between the latent representation and the input
        l - parameter of regularization lambda (L = L_reconstruct + \lambda L_RTD)
        """
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.decoder = copy.deepcopy(decoder)
        self.norm_constant = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.RTDLoss = RTDLoss
        self.MSELoss = MSELoss
        self.rtd_l = rtd_l
        self.rtd_every_n_batches = rtd_every_n_batches
        self.rtd_start_epoch = rtd_start_epoch
        self.lr = lr
    
    def forward(self, x):
        embedding = self.norm_constant * self.encoder(x)
        return embedding
    
    def z_dist(self, z):
        z_dist = torch.cdist(z, z)
#         if self.norm_constant is None:
#             self.norm_constant = 1.0 / np.quantile(z_dist.flatten().detach().cpu().numpy(), 0.9)
#         norm_constant = torch.quantile(z_dist.view(-1), 0.9)
        z_dist = self.norm_constant * (z_dist / np.sqrt(z_dist.shape[1]))
        return z_dist

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, x_dist, y = train_batch
        z = self.encoder(x)  
        x_hat = self.decoder(z)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('train/mse_loss', loss)
        if self.RTDLoss is not None:
            if (self.rtd_start_epoch <= self.current_epoch) and batch_idx % self.rtd_every_n_batches == 0:
                z_dist = self.z_dist(z)
                loss_xz, loss_zx, rtd_loss = self.RTDLoss(x_dist, z_dist)
                self.log('train/rtd_loss', rtd_loss)
                self.log('train/rtd_loss_xz', loss_xz)
                self.log('train/rtd_loss_zx', loss_zx)
                loss += self.rtd_l*rtd_loss
        self.log('train/loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_dist, y = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('val/mse_loss', loss)
        if self.RTDLoss is not None and self.rtd_start_epoch <= self.current_epoch+1:
            z_dist = self.z_dist(z)
            loss_xz, loss_zx, rtd_loss = self.RTDLoss(x_dist, z_dist)
            self.log('val/rtd_loss', rtd_loss)
            self.log('val/rtd_loss_xz', loss_xz)
            self.log('val/rtd_loss_zx', loss_zx)
            loss += self.rtd_l*rtd_loss
        self.log('val/loss', loss)
        
class DiagnosticAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, RTDLoss=None, MSELoss=None, rtd_l=0.1, rtd_every_n_batches=1, rtd_start_epoch=0, lr=5e-4, **kwargs):
        """
        RTDLoss - function of topological (RTD) loss between the latent representation and the input
        l - parameter of regularization lambda (L = L_reconstruct + \lambda L_RTD)
        """
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.decoder = copy.deepcopy(decoder)
        self.norm_constant = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.RTDLoss = RTDLoss
        self.MSELoss = MSELoss
        self.rtd_l = rtd_l
        self.rtd_every_n_batches = rtd_every_n_batches
        self.rtd_start_epoch = rtd_start_epoch
        self.lr = lr
    
    def forward(self, x):
        embedding = self.norm_constant * self.encoder(x)
        return embedding
    
    def z_dist(self, z):
        z_dist = torch.cdist(z, z)
#         if self.norm_constant is None:
#             self.norm_constant = 1.0 / np.quantile(z_dist.flatten().detach().cpu().numpy(), 0.9)
#         norm_constant = torch.quantile(z_dist.view(-1), 0.9)
        z_dist = self.norm_constant * (z_dist / np.sqrt(z_dist.shape[1]))
        return z_dist

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, x_dist, y = train_batch
        z = self.encoder(x)  
        x_hat = self.decoder(z)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('train/mse_loss', loss)
        if self.RTDLoss is not None:
            if (self.rtd_start_epoch <= self.current_epoch) and batch_idx % self.rtd_every_n_batches == 0:
                z_dist = self.z_dist(z)
                loss_xz, loss_zx, rtd_loss = self.RTDLoss(x_dist, z_dist)
                self.log('train/rtd_loss', rtd_loss)
                self.log('train/rtd_loss_xz', loss_xz)
                self.log('train/rtd_loss_zx', loss_zx)
                loss += self.rtd_l*rtd_loss
        self.log('train/loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_dist, y = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('val/mse_loss', loss)
        if self.RTDLoss is not None and self.rtd_start_epoch <= self.current_epoch+1:
            z_dist = self.z_dist(z)
            loss_xz, loss_zx, rtd_loss = self.RTDLoss(x_dist, z_dist)
            self.log('val/rtd_loss', rtd_loss)
            self.log('val/rtd_loss_xz', loss_xz)
            self.log('val/rtd_loss_zx', loss_zx)
            loss += self.rtd_l*rtd_loss
        self.log('val/loss', loss)
        return x, z, y
        
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
        
            