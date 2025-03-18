import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiLoss(nn.Module):
    def __init__(self, alpha, weight = False):
        # alpha: percentage of valence loss
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        mat_six = [0.8282945415389713, 0.8241984669908514, 1.0995052226498077, 3.4794711203897006, 1.4184397163120568, 4.657661853749418]
        mat_val = [0.4701457451810061, 0.8241984669908514, 1.0995052226498077]
        if weight:
            self.CE1 = nn.CrossEntropyLoss(weight = torch.tensor(mat_six).to(device))
            self.CE2 = nn.CrossEntropyLoss(weight = torch.tensor(mat_val).to(device))
        else:
            self.CE1 = nn.CrossEntropyLoss()
            self.CE2 = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.weight = weight
    

    def forward(self, label_six, label_v, pred_six, pred_v):
        mask = label_six >= 0
        loss_six = self.CE1(torch.masked_select(pred_six, mask.repeat(1, pred_six.shape[1])).reshape(-1, pred_six.shape[1]),
                       torch.masked_select(label_six, mask).view(-1))
        # loss_six = self.CE(pred_six,label_six.view(-1))
        mask = label_v >= 0
        loss_v = self.CE2(torch.masked_select(pred_v, mask.repeat(1, pred_v.shape[1])).reshape(-1, pred_v.shape[1]),
                     torch.masked_select(label_v, mask).view(-1))
        # loss_v = self.MSE(pred_v[~torch.isnan(label_v)],(label_v[~torch.isnan(label_v)]+1)/2)

        return loss_six * (1 - self.alpha) + loss_v * self.alpha