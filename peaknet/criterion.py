#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma


    def forward(self, batch_fmap_predicted, batch_mask_true):
        # Interpret multi-channel (multi-class) through softmax...
        batch_fmap_predicted = F.softmax(batch_fmap_predicted, dim = 1)    # B, C, H, W

        # Convert integer encoded mask into one-hot...
        batch_mask_true_onehot = self._create_one_hot(batch_mask_true)
        loss = self._calc_categorical_focal_loss(batch_fmap_predicted, batch_mask_true_onehot)

        return loss


    def _calc_categorical_focal_loss(self, y_pred, y):
        '''
        y_pred should be one-hot like.  y should use integer encoding.
        '''
        alpha = self.alpha
        gamma = self.gamma

        y_pred = y_pred.clamp(min=1e-8, max=1-1e-8)

        cross_entropy = -y * y_pred.log()
        loss = alpha * (1 - y_pred)**gamma * cross_entropy
        loss = loss.sum(dim = 1)    # sum across the class (one-hot) dimension

        return loss


    def _create_one_hot(self, batch_mask_true):
        '''
        B, C, H, W
        '''
        B, C, H, W = batch_mask_true.shape

        return F.one_hot(batch_mask_true.to(torch.long).reshape(B, -1)).permute(0, 2, 1).reshape(B, -1, H, W)
