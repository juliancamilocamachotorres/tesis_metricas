import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
import torch.nn.init
import random
import glob
from tqdm.notebook import tqdm
from pathlib import Path
import os
import math
import random
import itertools
import numpy as np
from scipy import io
from PIL import Image
from skimage import metrics
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm.notebook import tqdm
from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms.functional as TF
from os import listdir
from os.path import isfile, join
from torch.nn.modules.utils import _pair, _quadruple
from skimage import metrics
from PIL import Image
torch.cuda.empty_cache()

class levelsetLoss(nn.Module):
    def __init__(self):
        super(levelsetLoss, self).__init__()
    def forward(self, output, target):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:,ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2,3))/torch.sum(output)
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss
		
class gradientLoss2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss2d, self).__init__()
        self.penalty = penalty
    def forward(self, input):
        dH = torch.abs(input[ 1:, :] - input[ :-1, :])
        dW = torch.abs(input[ :, 1:] - input[ :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW
        loss = torch.sum(dH) + torch.sum(dW)
        return loss

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple ( ) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def __call__(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor

def get_center(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)
    x_map = torch.transpose(x_map,1,0)
    y_map = torch.transpose(y_map,1,0)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)

# Convolutional Segmentation Network
class CSNet(nn.Module):
    def __init__(self,input_dim):
        super(CSNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)
        #self.median_pool = MedianPool2d(kernel_size=5, stride=1, padding=0, same=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        #x = self.median_pool(x)
        return x

def get_variance(s_map, x_c, y_c):

    h,w = s_map.shape
    x_map, y_map = center.get_coordinate_tensors(h,w)
    x_map = torch.transpose(x_map,1,0)
    y_map = torch.transpose(y_map,1,0)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (s_map * v_x_map).sum()
    v_y = (s_map * v_y_map).sum()
    return v_x, v_y

def unir_mascaras(lista):
    if len(lista) == 12:
        concat_h1 = cv2.hconcat([lista[0], lista[1], lista[2], lista[3]])
        concat_h2 = cv2.hconcat([lista[4], lista[5], lista[6], lista[7]])
        concat_h3 = cv2.hconcat([lista[8], lista[9], lista[10], lista[11]])
        mascara_completa = cv2.vconcat([concat_h1, concat_h2, concat_h3])
    else :
        concat_h1 = cv2.hconcat([lista[0], lista[1], lista[2], lista[3], lista[4], lista[5]])
        concat_h2 = cv2.hconcat([lista[6], lista[7], lista[8], lista[9], lista[10], lista[11]])
        concat_h3 = cv2.hconcat([lista[12], lista[13], lista[14], lista[15], lista[16], lista[17]])
        concat_h4 = cv2.hconcat([lista[18], lista[19], lista[20], lista[21], lista[22], lista[23]])
        mascara_completa = cv2.vconcat([concat_h1, concat_h2, concat_h3, concat_h4])
    return mascara_completa


def calculate_precision(y_true, y_pred):
    """
    Calcula la precision de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      La precision.
    """

    intersection = np.sum((y_true & y_pred) == 1)
    union = np.sum((y_true | y_pred) == 1)

    if union == 0:
        return 0
    else:
        return intersection / union


def calculate_recall(y_true, y_pred):
    """
    Calcula el recall de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      El recall.
    """

    intersection = np.sum((y_true & y_pred) == 1)
    ground_truth = np.sum(y_true == 1)

    if ground_truth == 0:
        return 0
    else:
        return intersection / ground_truth

def calculate_metrics(y_true, y_pred):
    """
    Calcula el miou de dos matrices de segmentación.

    Args:
      y_true: La matriz de segmentación de referencia.
      y_pred: La matriz de segmentación predicha.

    Returns:
      El miou.
    """

    intersection = np.sum((y_true & y_pred) == 1)
    union = np.sum((y_true | y_pred) == 1)

    if union == 0:
        MIOU = 0
    else:
        MIOU = intersection / union
    
    """
    Calcula el SSIM de dos matrices de segmentación.

    Args:
      y_true: La matriz de segmentación de referencia.
      y_pred: La matriz de segmentación predicha.

    Returns:
      El SSIM.
    """

    SSIM = metrics.structural_similarity(y_true, y_pred)

    """
    Calcula el F1 de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      El F1.
    """

    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
    
    """
    Calcula el DICE de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      El DICE.
    """

    intersection = np.sum(y_true * y_pred)  # Calcular la intersección
    union = np.sum(y_true) + np.sum(y_pred)  # Calcular la unión

    if union == 0:
        DICE = 0
    else:
        DICE = (2 * intersection) / (union + 1e-12)

    return [MIOU, SSIM, F1, DICE]

class Args:
    def __init__(self):
        pass

if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        args = Args()

        args.nChannel = 100
        args.maxIter = 50
        args.minLabels = 3
        args.lr = 0.1
        args.nConv = 2
        args.stepsize_ce = 1
        args.stepsize_ss = 5
        args.center = False
        args.input = "/kaggle/working/"
        dataroot = Path('/kaggle/input/glasmiccai2015-gland-segmentation/Warwick_QU_Dataset')
        df = pd.read_csv('/kaggle/input/glascsv/data (1).csv')
        list_images = []
        #for i in list(df[df['fold'] ==2]['name']):
        #imgname = i +".bmp"
        #list_images.append(os.path.join(dataroot, imgname))
        pass
