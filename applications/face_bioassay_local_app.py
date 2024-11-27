#-*-coding:utf-8-*-
'''
DpCas-Light
||||      |||||        ||||         ||       |||||||
||  ||    ||   ||    ||    ||      ||||     ||     ||
||    ||  ||    ||  ||      ||    ||  ||     ||
||    ||  ||   ||   ||           ||====||     ||||||
||    ||  |||||     ||      ||  ||======||         ||
||  ||    ||         ||    ||  ||        ||  ||     ||
||||      ||           ||||   ||          ||  |||||||

/--------------------- Face Bioassay ---------------------/
'''
# date:2021-04-18
# Author: Eric.Lee
# function: Face Bioassay "基于人脸动作的活体检测"

import os
import cv2
import time

import numpy as np
import random
import time
import shutil

# 加载模型组件库
from face_detect.yolo_v3_face import yolo_v3_face_model
from face_multi_task.face_multi_task_component import FaceMuitiTask_Model
from face_euler_angle.face_euler_angle_component import FaceAngle_Model
from face_au.face_au_c import FaceAu_Model
# 加载工具库
import sys
sys.path.append("./lib/face_bioassay_lib/")
from cores.face_fuction import get_faces_batch_attribute
from utils.utils import parse_data_cfg

def main_face_bioassay(video_path,cfg_file):
    config = parse_data_cfg(cfg_file)
    print("\n/************** Face Bioassay *****************/")
    print("/********************************************************/\n")
    # pose_model = light_pose_model()
    face_detect_model = yolo_v3_face_model(conf_thres=float(config["detect_conf_thres"]),nms_thres=float(config["detect_nms_thres"]),
        model_arch = config["detect_model_arch"],model_path = config["detect_model_path"],yolo_anchor_scale = float(config["yolo_anchor_scale"]),
        img_size = float(config["detect_input_size"]),
        )
    face_multitask_model = FaceMuitiTask_Model(model_path = config["face_multitask_model_path"], model_arch = config["face_multitask_model_arch"])

    face_euler_model = FaceAngle_Model(model_path = config["face_euler_model_path"])

    face_au_model = FaceAu_Model(model_path = config["face_au_model_path"])

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret,img = cap.read()
        if ret:
            frame_idx += 1
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            faces_bboxes =face_detect_model.predict(img,vis = False) # 检测手，获取手的边界框

            faces_message = get_faces_batch_attribute(
                face_multitask_model,face_euler_model,face_au_model,faces_bboxes,img,use_cuda = True,vis = True)
            if faces_message is not None:
                print("faces_message : {} \n".format(faces_message))
                pass

            cv2.namedWindow("DriverFatigueMonitor",0)
            cv2.imshow("DriverFatigueMonitor",img)
            if cv2.waitKey(1)==27:
                break
        else:
            break

    cv2.destroyAllWindows()
