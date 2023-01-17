import torch
from torch import nn
from .utils import torch_version_major, torch_version_minor
import networkx as nx
import malis.malis as m
from scipy.ndimage.morphology import binary_dilation as bind
from skimage import measure
import numpy as np


class MSELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weights=None):
        loss = (pred-target).pow(2)
        
        if weights is not None:
            loss *= weights

        #if self.ignore_index is not None:
        #    loss = loss[target!=self.ignore_index]

        return loss.mean()

class MSELoss2Proj(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.mse1 = MSELoss(ignore_index=ignore_index)
        self.mse2 = MSELoss(ignore_index=ignore_index)
        self.mse3 = MSELoss(ignore_index=ignore_index)

    def forward(self, pred, target, weights=[None,None,None]):
        # 0- batch, 1- channel, 2- height, 3- width, 4- depth
        projection1,_ = pred.min(2)
        projection1=projection1.reshape(pred.size(0),pred.size(1),
                                        pred.size(3),pred.size(4))
        projection2,_ = pred.min(3)
        projection2=projection2.reshape(pred.size(0),pred.size(1),
                                        pred.size(2),pred.size(4))

        l1,_ = target.min(2)
        l1 = l1.reshape(target.size(0),target.size(1),
                        target.size(3),target.size(4))
        l2,_ = target.min(3)
        l2 = l2.reshape(target.size(0),target.size(1),
                        target.size(2),target.size(4))

        loss1=self.mse1(projection1,l1,weights[0])
        loss2=self.mse2(projection2,l2,weights[0])
        return loss1+loss2

class MSELoss3Proj(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.mse1 = MSELoss(ignore_index=ignore_index)
        self.mse2 = MSELoss(ignore_index=ignore_index)
        self.mse3 = MSELoss(ignore_index=ignore_index)

    def forward(self, pred, target, weights=[None,None,None]):
        # 0- batch, 1- channel, 2- height, 3- width, 4- depth
        projection1,_ = pred.min(2)
        projection1=projection1.reshape(pred.size(0),pred.size(1),
                                        pred.size(3),pred.size(4))
        projection2,_ = pred.min(3)
        projection2=projection2.reshape(pred.size(0),pred.size(1),
                                        pred.size(2),pred.size(4))
        projection3,_ = pred.min(4)
        projection3=projection3.reshape(pred.size(0),pred.size(1),
                                        pred.size(2),pred.size(3))

        l1,_ = target.min(2)
        l1 = l1.reshape(target.size(0),target.size(1),
                        target.size(3),target.size(4))
        l2,_ = target.min(3)
        l2 = l2.reshape(target.size(0),target.size(1),
                        target.size(2),target.size(4))
        l3,_ = target.min(4)
        l3 = l3.reshape(target.size(0),target.size(1),
                        target.size(2),target.size(3))

        loss1=self.mse1(projection1,l1,weights[0])
        loss2=self.mse2(projection2,l2,weights[0])
        loss3=self.mse3(projection3,l3,weights[0])
        return loss1+loss2+loss3

class ConnLoss3Proj(nn.Module):

    def __init__(self, dmax=15, ignore_index=255):
        super().__init__()
        self.malis1 = MALISLoss_window_pos(dmax=dmax, ignore_index=ignore_index)
        self.malis2 = MALISLoss_window_pos(dmax=dmax, ignore_index=ignore_index)
        self.malis3 = MALISLoss_window_pos(dmax=dmax, ignore_index=ignore_index)
        
    def forward(self, pred, target, malis_lr, malis_lr_pos, weights=[None,None,None]):
        # 0- batch, 1- channel, 2- height, 3- width, 4- depth
        projection1,_ = pred.min(2)
        projection1=projection1.reshape(pred.size(0),pred.size(1),
                                        pred.size(3),pred.size(4))
        projection2,_ = pred.min(3)
        projection2=projection2.reshape(pred.size(0),pred.size(1),
                                        pred.size(2),pred.size(4))
        projection3,_ = pred.min(4)
        projection3=projection3.reshape(pred.size(0),pred.size(1),
                                        pred.size(2),pred.size(3))
        
        
        l1,_ = target.min(2)
        l1 = l1.reshape(target.size(0),target.size(1),
                        target.size(3),target.size(4))
        l2,_ = target.min(3)
        l2 = l2.reshape(target.size(0),target.size(1),
                        target.size(2),target.size(4))
        l3,_ = target.min(4)
        l3 = l3.reshape(target.size(0),target.size(1),
                        target.size(2),target.size(3))

        loss1=self.malis1(projection1,l1,malis_lr,malis_lr_pos,weights[0])
        loss2=self.malis2(projection2,l2,malis_lr,malis_lr_pos,weights[1])
        loss3=self.malis3(projection3,l3,malis_lr,malis_lr_pos,weights[2])
        return loss1+loss2+loss3

class ConnLoss2Proj(nn.Module):

    def __init__(self, dmax=15, ignore_index=255):
        super().__init__()
        self.malis1 = MALISLoss_window_pos(dmax=dmax, ignore_index=ignore_index)
        self.malis2 = MALISLoss_window_pos(dmax=dmax, ignore_index=ignore_index)
        
    def forward(self, pred, target, malis_lr, malis_lr_pos, weights=[None,None,None]):
        # 0- batch, 1- channel, 2- height, 3- width, 4- depth
        projection1,_ = pred.min(2)
        projection1=projection1.reshape(pred.size(0),pred.size(1),
                                        pred.size(3),pred.size(4))
        projection2,_ = pred.min(3)
        projection2=projection2.reshape(pred.size(0),pred.size(1),
                                        pred.size(2),pred.size(4))

        l1,_ = target.min(2)
        l1 = l1.reshape(target.size(0),target.size(1),
                        target.size(3),target.size(4))
        l2,_ = target.min(3)
        l2 = l2.reshape(target.size(0),target.size(1),
                        target.size(2),target.size(4))

        loss1=self.malis1(projection1,l1,malis_lr,malis_lr_pos,weights[0])
        loss2=self.malis2(projection2,l2,malis_lr,malis_lr_pos,weights[1])
        return loss1+loss2

class MALISLoss_window_pos(nn.Module):

    def __init__(self, dmax=15, ignore_index=255):
        super().__init__()
        self.dmax = dmax
        self.ignore_index = ignore_index

    def forward(self, pred, target, malis_lr, malis_lr_pos, weights=None):
        pred_np_full = pred.cpu().detach().numpy()
        target_np_full = target.cpu().detach().numpy()
        B,C,H,W = pred_np_full.shape

        weights_n = np.zeros(pred_np_full.shape)
        weights_p = np.zeros(pred_np_full.shape)
        window = 48

        for k in range(H // window):
            for j in range(W // window):
                pred_np = pred_np_full[:,:,k*window:(k+1)*window,j*window:(j+1)*window]
                target_np = target_np_full[:,:,k*window:(k+1)*window,j*window:(j+1)*window]

                nodes_indexes = np.arange(window*window).reshape(window,window)
                nodes_indexes_h = np.vstack([nodes_indexes[:,:-1].ravel(), nodes_indexes[:,1:].ravel()]).tolist()
                nodes_indexes_v = np.vstack([nodes_indexes[:-1,:].ravel(), nodes_indexes[1:,:].ravel()]).tolist()
                nodes_indexes = np.hstack([nodes_indexes_h, nodes_indexes_v])
                nodes_indexes = np.uint64(nodes_indexes)

                costs_h = (pred_np[:,:,:,:-1] + pred_np[:,:,:,1:]).reshape(B,-1)
                costs_v = (pred_np[:,:,:-1,:] + pred_np[:,:,1:,:]).reshape(B,-1)
                costs = np.hstack([costs_h, costs_v])
                costs = np.float32(costs)

                gtcosts_h = (target_np[:,:,:,:-1] + target_np[:,:,:,1:]).reshape(B,-1)
                gtcosts_v = (target_np[:,:,:-1,:] + target_np[:,:,1:,:]).reshape(B,-1)
                gtcosts = np.hstack([gtcosts_h, gtcosts_v])
                gtcosts = np.float32(gtcosts)

                costs_n = costs.copy()
                costs_p = costs.copy()

                costs_n[gtcosts > 15] = 15
                costs_p[gtcosts < 8] = 0
                gtcosts[gtcosts > 15] = 15

                for i in range(len(pred_np)):
                    sg_gt = measure.label(bind((target_np[i,0] == 0), iterations=5)==0)

                    edge_weights_n = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], \
                                           nodes_indexes[1], costs_n[i], 0)

                    edge_weights_p = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], \
                                           nodes_indexes[1], costs_p[i], 1)


                    num_pairs_n = np.sum(edge_weights_n)
                    if num_pairs_n > 0:
                        edge_weights_n = edge_weights_n/num_pairs_n

                    num_pairs_p = np.sum(edge_weights_p)
                    if num_pairs_p > 0:
                        edge_weights_p = edge_weights_p/num_pairs_p

                    edge_weights_n[gtcosts[i] >= 8] = 0
                    edge_weights_p[gtcosts[i] < 15] = 0

                    malis_w = edge_weights_n.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window,window-1), malis_w_v.reshape(window-1,window)

                    nodes_weights = np.zeros((window,window), np.float32)
                    nodes_weights[:,:-1] += malis_w_h
                    nodes_weights[:,1:] += malis_w_h
                    nodes_weights[:-1,:] += malis_w_v
                    nodes_weights[1:,:] += malis_w_v

                    weights_n[i, 0, k*window:(k+1)*window, j*window:(j+1)*window] = nodes_weights

                    malis_w = edge_weights_p.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window,window-1), malis_w_v.reshape(window-1,window)

                    nodes_weights = np.zeros((window,window), np.float32)
                    nodes_weights[:,:-1] += malis_w_h
                    nodes_weights[:,1:] += malis_w_h
                    nodes_weights[:-1,:] += malis_w_v
                    nodes_weights[1:,:] += malis_w_v

                    weights_p[i, 0, k*window:(k+1)*window, j*window:(j+1)*window] = nodes_weights

        loss_n = (pred).pow(2)
        loss_p = (self.dmax - pred).pow(2)
        loss = malis_lr * loss_n * torch.Tensor(weights_n).cuda() + malis_lr_pos * loss_p * torch.Tensor(weights_p).cuda()

        return loss.sum()
