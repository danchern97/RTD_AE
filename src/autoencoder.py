#!/usr/bin/python3
import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

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
        embedding = self.encoder(x)
        return embedding
    
    def z_dist(self, z):
        z_dist = torch.cdist(z, z)
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