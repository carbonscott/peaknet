#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, num_classes = 3):
        super().__init__()

        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.num_classes = num_classes

        self.min_clamp = 1e-8
        self.max_clamp = 1-1e-8


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
        # Move alpha to device if it hasn't already???
        if self.alpha.device != y_pred.device:
            self.alpha = self.alpha.to(y_pred.device)

        alpha       = self.alpha
        gamma       = self.gamma
        num_classes = self.num_classes

        y_pred = y_pred.clamp(min=self.min_clamp, max=self.max_clamp)

        cross_entropy = -y * y_pred.log()
        loss = alpha.view(1, num_classes, 1, 1) * (1 - y_pred)**gamma * cross_entropy
        loss = loss.sum(dim = 1)    # sum across the class (one-hot) dimension

        return loss


    def _create_one_hot(self, batch_mask_true):
        '''
        B, C, H, W
        '''
        B, C, H, W = batch_mask_true.shape

        return F.one_hot(batch_mask_true.to(torch.long).view(B, -1), num_classes = self.num_classes).permute(0, 2, 1).view(B, -1, H, W)
