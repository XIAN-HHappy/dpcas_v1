#-*-coding:utf-8-*-
# date:2021-03-09
# Author: Eric.Lee
# function: handpose_x 21 keypoints 2D

import os
import torch
import cv2
import numpy as np
import json

import torch
import torch.nn as nn

import time
import math
from datetime import datetime

from face_au.models.resnet import resnet18, resnet34, resnet50, resnet101
from face_au.models.mobilenetv2 import MobileNetV2

#
class FaceAu_Model(object):
    def __init__(self,
        model_path = './components/face_au/weights/face_au-resnet50-size256-20210427.pth',
        img_size= 256,
        num_classes = 24,
        model_arch = "resnet_50",
        ):
        # print("face au loading : ",model_path)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu") # 可选的设备类型及序号
        self.img_size = img_size
        #-----------------------------------------------------------------------

        if model_arch == 'resnet_50':
            model_ = resnet50(num_classes = num_classes,img_size = self.img_size)
        elif model_arch == 'resnet_18':
            model_ = resnet18(num_classes = num_classes,img_size = self.img_size)
        elif model_arch == 'resnet_34':
            model_ = resnet34(num_classes = num_classes,img_size = self.img_size)
        elif model_arch == 'resnet_101':
            model_ = resnet101(num_classes = num_classes,img_size = self.img_size)

        #-----------------------------------------------------------------------
        model_ = model_.to(self.device)
        # 加载测试模型
        if os.access(model_path,os.F_OK):# checkpoint
            chkpt = torch.load(model_path, map_location=self.device)
            model_.load_state_dict(chkpt)
            print('face au model loading : {}'.format(model_path))

        model_.eval() # 设置为前向推断模式
        self.model_au = model_

    def predict(self, img, vis = False):
        with torch.no_grad():
            # img_ = img_ - [123.67, 116.28, 103.53]
            img_ = torch.from_numpy(img)
            # img_ = img_.unsqueeze_(0)
            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            output_ = self.model_au(img_.float())
            # print(pre_.size())
            output_ = output_.cpu().detach().numpy()

        return output_
