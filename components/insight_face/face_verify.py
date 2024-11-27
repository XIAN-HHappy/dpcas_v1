#-*-coding:utf-8-*-
# date:2021-04-16
# Author: Eric.Lee
# function: face verify

import warnings
warnings.filterwarnings("ignore")
import os
import torch
from insight_face.model import Backbone,MobileFaceNet
from insight_face.utils import load_facebank,infer
from pathlib import Path
from PIL import Image
import cv2

class insight_face_model(object):
    def __init__(self,
        net_mode = "ir_se", # [ir, ir_se, mobilefacenet]
        net_depth = 50, # [50,100,152]
        backbone_model_path = "./components/insight_face/weights/model_ir_se-50.pth",
        facebank_path = "./components/insight_face/facebank", # 人脸比对底库
        tta = False,
        threshold = 1.2 ,
        embedding_size = 512,
        ):

        self.threshold = threshold
        self.tta = tta
        device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if net_mode == "mobilefacenet":
            model_ = MobileFaceNet(embedding_size).to(device_)
            print('MobileFaceNet model generated')
        else:
            model_ = Backbone(net_depth, 1., net_mode).to(device_)
            print('{}_{} model generated'.format(net_mode, net_depth))

        if os.access(backbone_model_path,os.F_OK):
            model_.load_state_dict(torch.load(backbone_model_path))
            print("-------->>>   load model : {}".format(backbone_model_path))

        model_.eval()
        self.model_ = model_
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #------------------- 加载人脸比对底库
        targets, names =  load_facebank(facebank_path)

        self.face_targets = targets
        self.face_names = names

        print("faces verify names : \n {}".format(self.face_names))
        print("targets size : {}".format(self.face_targets.size()))

    def predict(self, faces_identify, vis = False):
        with torch.no_grad():

            results, face_dst = infer(self.model_, self.device_, faces_identify, self.face_targets, threshold = self.threshold ,tta=self.tta)
            # print(results, face_dst)

        return results, face_dst
    # print("names : {}".format(names))
    # print("targets size : {}".format(targets.size()))
    #
    # #---------------------------------------------------------------------------
    # if True:
    #     print("\n---------------------------\n")
    #     faces_identify = []
    #     idx = 0
    #     for file in os.listdir(args.example):
    #         img = cv2.imread(args.example + file) # 图像必须 112*112
    #         faces_identify.append(Image.fromarray(img))
    #
    #         results, face_dst = infer(model_, device_, faces_identify, targets, threshold = 1.2 ,tta=False)
    #
    #         face_dst = list(face_dst.cpu().detach().numpy())
    #
    #         print("{}) recognize：{} ,dst : {}".format(idx+1,names[results[idx] + 1],face_dst[idx]))
    #
    #         cv2.putText(img, names[results[idx] + 1], (2,13),cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 0, 220),5)
    #         cv2.putText(img, names[results[idx] + 1], (2,13),cv2.FONT_HERSHEY_DUPLEX, 0.38, (255, 50, 50),1)
    #
    #         cv2.namedWindow("imag_face",0)
    #         cv2.imshow("imag_face",img)
    #         cv2.waitKey(0)
    #
    #         idx += 1
    #     cv2.destroyAllWindows()
    # else:
    #     #---------------------------------------------------------------------------
    #     print("\n---------------------------\n")
    #     faces_identify = []
    #     idx = 0
    #     sum = 0
    #     r_ = 0
    #     for doc_ in os.listdir(args.example):
    #         for file in os.listdir(args.example + doc_):
    #             img = cv2.imread(args.example + doc_ + "/" +  file) # 图像必须 112*112
    #             faces_identify.append(Image.fromarray(img))
    #
    #             results, face_dst = infer(model_, device_, faces_identify, targets, threshold = 1.2 ,tta=False)
    #
    #             face_dst = list(face_dst.cpu().detach().numpy())
    #
    #             print("{}) gt : {} ~ recognize：{} , dst : {}".format(idx+1,doc_,names[results[idx] + 1],face_dst[idx]))
    #
    #             #
    #             sum += 1
    #             if doc_ == names[results[idx] + 1]:
    #                 r_ += 1
    #             print("     {}- {}  -->> precision ： {}".format(r_,sum,r_/sum))
    #
    #             idx += 1
    #
    #             cv2.namedWindow("imag_face",0)
    #             cv2.imshow("imag_face",img)
    #             cv2.waitKey(1)
    #     cv2.destroyAllWindows()
