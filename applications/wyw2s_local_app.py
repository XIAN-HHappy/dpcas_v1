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

/--------------------- Who You Want To See ---------------------/
'''
# date:2021-04-18
# Author: Eric.Lee
# function: who you want to see "你想看谁"

import os
import cv2
import time

from multiprocessing import Process
from multiprocessing import Manager

import numpy as np
import random
import time
import shutil

# 加载模型组件库
from face_detect.yolo_v3_face import yolo_v3_face_model
from insight_face.face_verify import insight_face_model
from face_multi_task.face_multi_task_component import FaceMuitiTask_Model
from face_euler_angle.face_euler_angle_component import FaceAngle_Model
# 加载工具库
import sys
sys.path.append("./lib/wyw2s_lib/")
from cores.wyw2s_fuction import get_faces_batch_attribute
from utils.utils import parse_data_cfg
from utils.show_videos_thread import run_show
from moviepy.editor import *


def main_wyw2s(video_path,cfg_file):

    config = parse_data_cfg(cfg_file)

    print("\n/---------------------- main_wyw2s config ------------------------/\n")
    for k_ in config.keys():
        print("{} : {}".format(k_,config[k_]))
    print("\n/------------------------------------------------------------------------/\n")

    print("\n loading who you want 2 see local demo ...\n")

    face_detect_model = yolo_v3_face_model(conf_thres=float(config["detect_conf_thres"]),nms_thres=float(config["detect_nms_thres"]),
        model_arch = config["detect_model_arch"],model_path = config["detect_model_path"],yolo_anchor_scale = float(config["yolo_anchor_scale"]),
        img_size = float(config["detect_input_size"]),
        )
    face_verify_model = insight_face_model(backbone_model_path =config["face_verify_backbone_path"] ,
        facebank_path = config["facebank_path"],
        threshold = float(config["face_verify_threshold"]))

    face_multitask_model = FaceMuitiTask_Model(model_path = config["face_multitask_model_path"], model_arch = config["face_multitask_model_arch"])

    face_euler_model = FaceAngle_Model(model_path = config["face_euler_model_path"])

    print("\n/------------------------------------------------------------------------/\n")
    YouWantToSee = config["YouWantToSee"]
    YouWantToSee_=[name_ for name_ in YouWantToSee.split(",")]
    print("  YouWantToSee : {}".format(YouWantToSee_))
    print("\n/------------------------------------------------------------------------/\n")

    p_colors = []
    for i in range(len(face_verify_model.face_names)):
        if i == 0 :
            p_colors.append((100,155,100))
        if i == 1 :
            p_colors.append((0,255,0))
        elif i == 2:
            p_colors.append((255,0,0))
        elif i == 3:
            p_colors.append((0,255,255))
        elif i == 4:
            p_colors.append((0,185,255))
        elif i == 5:
            p_colors.append((255,185,55))
        else:
            p_colors.append((random.randint(60,255),random.randint(70,255),random.randint(130,255)))

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 时间轴
    time_map = np.zeros([200,frame_count,3]).astype(np.uint8)
    time_map[:,:,0].fill(105)
    time_map[:,:,1].fill(105)
    time_map[:,:,2].fill(105)

    pts_last = None
    frame_idx = 0
    Flag_Last,Flag_Now = False,False
    start_time = 0.
    YouWantToSee_time_list = []

    while cap.isOpened():
        ret,img = cap.read()
        if ret:
            frame_idx += 1
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            algo_image = img.copy()

            faces_bbox =face_detect_model.predict(img,vis = True) # 检测手，获取手的边界框
            if len(faces_bbox) > 0:
                faces_identify,faces_identify_bboxes,bboxes,face_map = get_faces_batch_attribute(face_multitask_model,face_euler_model,faces_bbox,algo_image,use_cuda = True,vis = True)
                YouHaveSeen_list = []
                if len(faces_identify) > 0:
                    results, face_dst = face_verify_model.predict(faces_identify)
                    face_dst = list(face_dst.cpu().detach().numpy())
                    # print("face_dst : ",face_dst)

                    for idx,bbox_ in enumerate(faces_identify_bboxes):

                        cv2.putText(algo_image, "{}: {:.2f}".format(face_verify_model.face_names[results[idx] + 1],face_dst[idx]), (bbox_[0],bbox_[1]+23),cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 95,220), 5)
                        cv2.putText(algo_image, "{}: {:.2f}".format(face_verify_model.face_names[results[idx] + 1],face_dst[idx]), (bbox_[0],bbox_[1]+23),cv2.FONT_HERSHEY_DUPLEX, 0.7, p_colors[results[idx] + 1], 2)
                        if face_verify_model.face_names[results[idx] + 1] in YouWantToSee_:
                            YouHaveSeen_list.append(face_verify_model.face_names[results[idx] + 1])
                # 绘制时间轴
                if len(YouHaveSeen_list)>0:
                    cv2.rectangle(time_map, (frame_idx-1,0), (frame_idx,100), (0,255,0), -1) # 绘制时间轴
                    print("  YouHaveSeen : {}".format(YouHaveSeen_list))
                    Flag_Now = True
                else:
                    cv2.rectangle(time_map, (frame_idx-1,100), (frame_idx,200), (255,0,0), -1) # 绘制时间轴
                    print("  ------ ")
                    Flag_Now = False

            else:
                face_map = np.zeros([112*3,112*3,3]).astype(np.uint8)
                face_map[:,:,0].fill(205)
                face_map[:,:,1].fill(205)
                face_map[:,:,2].fill(205)
                print(" ------ ")
                Flag_Now = False

                cv2.rectangle(time_map, (frame_idx-1,100), (frame_idx,200), (255,0,0), -1) # 绘制时间轴
            cv2.line(time_map, (frame_idx,100),(frame_idx,100), (0,80,255), 8) # 绘制时间轴 中轴线

            #-------------
            if Flag_Now == True and Flag_Last == False:
                start_time = video_time
            elif Flag_Now == False and Flag_Last == True:
                YouWantToSee_time_list.append((start_time/1000.,video_time/1000.))

            Flag_Last = Flag_Now
            #
            cv2.putText(algo_image, "WhoYouWant 2 See", (algo_image.shape[1]-420,45),cv2.FONT_HERSHEY_DUPLEX, 1.2, (205, 95,250), 7)
            cv2.putText(algo_image, "WhoYouWant 2 See", (algo_image.shape[1]-420,45),cv2.FONT_HERSHEY_DUPLEX, 1.2, (12, 255,12), 2)

            cv2.putText(algo_image, "DpCas -", (algo_image.shape[1]-620,45),cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 95,210), 7)
            cv2.putText(algo_image, "DpCas -", (algo_image.shape[1]-620,45),cv2.FONT_HERSHEY_DUPLEX, 1.2, (12, 255,12), 2)

            cv2.rectangle(algo_image, (algo_image.shape[1]-640,5), (algo_image.shape[1]-30,65), (0,185,255), 6)
            cv2.rectangle(algo_image, (algo_image.shape[1]-640,5), (algo_image.shape[1]-30,65), (255,100,100), 2)

            cv2.putText(algo_image, "[{}/{}]".format(frame_idx,frame_count), (5,30),cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 95,220), 5)
            cv2.putText(algo_image, "[{}/{}]".format(frame_idx,frame_count), (5,30),cv2.FONT_HERSHEY_DUPLEX, 1.0, (12, 255,12), 2)

            cv2.putText(algo_image, "[{:.2f} sec]".format(video_time/1000.), (5,70),cv2.FONT_HERSHEY_DUPLEX, 1.0, (15, 185,255), 7)
            cv2.putText(algo_image, "[{:.2f} sec]".format(video_time/1000.), (5,70),cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 112,112), 2)

            #
            cv2.putText(face_map, "YouWantToSee {}".format(YouWantToSee_), (5,face_map.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 95,220), 5)
            cv2.putText(face_map, "YouWantToSee {}".format(YouWantToSee_), (5,face_map.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.55, (12, 215,12), 1)

            face_map = cv2.resize(face_map,(algo_image.shape[0],algo_image.shape[0]))
            algo_image = np.hstack((algo_image,face_map)) # 合并显示
            time_map_r = cv2.resize(time_map,(algo_image.shape[1],200))
            algo_image = np.vstack((algo_image,time_map_r))

            cv2.namedWindow('WhoYouWant2See', 0)
            cv2.imshow('WhoYouWant2See', algo_image)

            key_id = cv2.waitKey(1)
            if key_id == 27:
                break
        else:
            break
        #-------------
    print("\n ----->>> YouWantToSee_Time_list : \n")
    movie = VideoFileClip(video_path)
    video_s = "./clip_wyw2s/"

    if os.path.exists(video_s): #删除之前的文件夹
        shutil.rmtree(video_s)

    if not os.path.exists(video_s): # 如果文件夹不存在
        os.mkdir(video_s) # 生成文件夹

    seg_idx = 0
    for seg_ in YouWantToSee_time_list:
        seg_idx += 1
        print(" Seg {} : {}".format(seg_idx,seg_))
        print(" 开始剪切目标人物视频 第 {} 段 \n".format(seg_idx))
        movie_clip = movie.subclip(seg_[0],seg_[1])# 将剪切的片段保存
        movie_clip.write_videofile("{}clip_{}.mp4".format(video_s,seg_idx))

    run_show(path = video_s , vis = True)

    cv2.destroyAllWindows()
