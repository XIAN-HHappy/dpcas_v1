#-*-coding:utf-8-*-
# date:2020-04-11
# author: Eric.Lee
# function : face_euler_angle - yaw,pitch,roll

import os
import torch
import cv2
import torch.nn.functional as F

from face_euler_angle.network.resnet import resnet18
from face_euler_angle.utils.common_utils import *

import numpy as np
class FaceAngle_Model(object):
    def __init__(self,
        model_path = './components/face_euler_angle/weights_euler_angle/resnet_18_imgsize_256-epoch-225.pth',
        img_size=256,
        num_classes = 3,# yaw,pitch,roll
        ):

        use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if use_cuda else "cpu") # 可选的设备类型及序号
        self.img_size = img_size
        #-----------------------------------------------------------------------
        model_ = resnet18(num_classes=num_classes, img_size=img_size)
        chkpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_.load_state_dict(chkpt)
        model_.eval()

        print("load face euler angle model : {}".format(model_path))

        self.model_ = model_.to(self.device)

    def predict(self, img,vis = False):# img is align img
        with torch.no_grad():
            img_ = torch.from_numpy(img)
            # img_ = img_.unsqueeze_(0)
            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            output_ = self.model_(img_.float())
            # print(pre_.size())
            output_ = output_.cpu().detach().numpy()


            output_ = output_*90.

        return output_
