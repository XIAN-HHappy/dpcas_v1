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

/-------------------- APP_X --------------------/
'''
# date:2020-10-19.7.23.24
# Author: DataBall - XIAN
# function: main

import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./components/") # 添加模型组件路径

def demo_logo():
    print("\n/*********************************/")
    print("/---------------------------------/\n")
    print("       WELCOME : DpCas-Light      ")
    print("           << APP_X >>         ")
    print("    Copyright 2021 Eric.Lee2021   ")
    print("        Apache License 2.0       ")
    print("\n/---------------------------------/")
    print("/*********************************/\n")

if __name__ == '__main__':
    demo_logo()
    parser = argparse.ArgumentParser(description= " DpCas : << Deep Learning Componentized Application System >> ")
    parser.add_argument('-app', type=int, default = 0,
        help = "handpose_x:0, gesture:1 ,wyw2s:2, face_bioassay:3, video_ana:4, face_pay:5, drive:6") # 设置 App Example

    app_dict = {
        0:"handpose_x",
        1:"gesture",
        2:"wyw2s",
        3:"face_bioassay",
        4:"video_ana",
        5:"face_pay",
        6:"drive"}

    args = parser.parse_args()# 解析添加参数

    APP_P = app_dict[args.app]

    if APP_P == "handpose_x": # 手势识别
        from applications.handpose_local_app import main_handpose_x #加载 handpose 应用
        cfg_file = "./lib/hand_lib/cfg/handpose.cfg"
        main_handpose_x(cfg_file)#加载 handpose 应用
    elif APP_P == "gesture": # 手势识别
        from applications.gesture_local_app import main_gesture_x #加载 gesture 应用
        cfg_file = "./lib/gesture_lib/cfg/handpose.cfg"
        main_gesture_x(cfg_file)#加载 handpose 应用
    elif APP_P == "wyw2s": # 基于人脸识别的视频剪辑
        from applications.wyw2s_local_app import main_wyw2s #加载 who you want 2 see 应用
        cfg_file = "./lib/wyw2s_lib/cfg/wyw2s.cfg"
        main_wyw2s(video_path = "./video/rw_11.mp4",cfg_file = cfg_file)#加载 who you want 2 see  应用

    elif APP_P == "face_bioassay":
        from applications.face_bioassay_local_app import main_face_bioassay #face_bioassay 应用
        cfg_file = "./lib/face_bioassay_lib/cfg/face_bioassay.cfg"
        video_path  =  "./video/face2.mp4"
        main_face_bioassay(video_path = video_path,cfg_file = cfg_file)#加载 face_bioassay  应用

    # elif APP_P == "video_ana":
    #     from applications.VideoAnalysis_app import main_VideoAnalysis #加载 video_analysis 应用
    #     main_VideoAnalysis(video_path = "./video/f3.mp4")#加载 video_analysis  应用
    #
    # elif APP_P == "face_pay":
    #     cfg_file = "./lib/facepay_lib/cfg/facepay.cfg"
    #     from applications.FacePay_local_app import main_facePay #加载 face pay 应用
    #     main_facePay(video_path = 0,cfg_file = cfg_file) # 加载 face pay  应用
    #
    # elif APP_P == "drive":
    #     from applications.DangerousDriveWarning_local_app import main_DangerousDriveWarning #加载 危险驾驶预警 应用
    #     cfg_file = "./lib/dfmonitor_lib/cfg/dfm.cfg"
    #     main_DangerousDriveWarning(video_path = "./video/drive1.mp4",cfg_file = cfg_file)

    print(" well done ~")
