#-*-coding:utf-8-*-
# date:2020-04-11
# author: Eric.Lee
# function : face_multi_task - landmarks & age & gender

import os
import torch
import cv2
import torch.nn.functional as F

from face_multi_task.network.resnet import resnet50,resnet34,resnet18
from face_multi_task.utils.common_utils import *

import numpy as np
class FaceMuitiTask_Model(object):
    def __init__(self,
        model_path = './components/face_multi_task/weights_multask/resnet_50_imgsize-256-20210411.pth',
        img_size=256,
        num_classes = 196,# 人脸关键点，年龄，性别
        model_arch = "resnet50",# 模型结构
        ):

        use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if use_cuda else "cpu") # 可选的设备类型及序号
        self.img_size = img_size
        #-----------------------------------------------------------------------
        if model_arch == "resnet50":
            face_multi_model = resnet50(landmarks_num=num_classes, img_size=img_size)
        elif model_arch == "resnet34":
            face_multi_model = resnet34(landmarks_num=num_classes, img_size=img_size)
        elif model_arch == "resnet18":
            face_multi_model = resnet18(landmarks_num=num_classes, img_size=img_size)

        chkpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        face_multi_model.load_state_dict(chkpt)
        face_multi_model.eval()

        print("load face multi task model : {}".format(model_path))

        self.face_multi_model = face_multi_model.to(self.device)

    def predict(self, img,vis = False):# img is align img
        with torch.no_grad():
            # img = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_LINEAR)
            # #-------------------- inferece face
            #
            # img_ = img.astype(np.float32)
            # img_ = (img_-128.)/256.
            #
            # img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img)
            # img_ = img_.unsqueeze_(0)
            #
            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            output_landmarks,output_gender,output_age = self.face_multi_model(img_.float())
            # print(pre_.size())
            output_landmarks = output_landmarks.cpu().detach().numpy()

            output_gender = output_gender
            output_gender = output_gender.cpu().detach().numpy()
            output_gender = np.array(output_gender)

            output_age = output_age.cpu().detach().numpy()
            output_age = (output_age*100.+50.)

        return output_landmarks,output_gender,output_age
if __name__ == '__main__':
    au_model = FaceMuitiTask_Component()
    path = "./samples/"
    for img_name in os.listdir(path):
        img_path  =  path + img_name
        img = cv2.imread(img_path)
        dict_landmarks,output_gender,output_age = au_model.predict(img,vis = False)
        draw_contour(img,dict_landmarks)


        cv2.putText(img, 'gender:{}'.format(output_gender), (2,20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
        cv2.putText(img, 'gender:{}'.format(output_gender), (2,20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,20,0),1)

        cv2.putText(img, 'age:{:.2f}'.format(output_age), (2,50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
        cv2.putText(img, 'age:{:.2f}'.format(output_age), (2,50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,20, 0),1)

        cv2.namedWindow('image',0)
        cv2.imshow('image',img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
