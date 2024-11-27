#-*-coding:utf-8-*-
'''
DpCas-Light
||||      ||||       ||||         ||        ||||
||  ||    ||  ||   ||    ||      ||||     ||    ||
||    ||  ||   || ||      ||    ||  ||     ||
||    ||  ||  ||  ||           ||====||      ||||
||    ||  ||||    ||      ||  ||======||         ||
||  ||    ||       ||    ||  ||        ||  ||    ||
||||      ||         ||||   ||          ||   ||||

/------------------ Who You Want 2 See ------------------/
'''
# date:2020-12-12
# Author: Eric.Lee
# function: show clip video
import os
import cv2
import copy
import time

import threading
from threading import current_thread, Lock

import  psutil
import numpy as np
import random

lock = Lock()
def run_one_process(path,process_id,vis):
    
    lock.acquire()
    video_ = cv2.VideoCapture(path)
    lock.release()
    while True:
        ret, img_ = video_.read()

        if ret:
           #------------------------------------------------
            if vis:
                cv2.namedWindow('video_seg_{}'.format(process_id),0)
                cv2.resizeWindow('video_seg_{}'.format(process_id), 300, 210);
                cv2.moveWindow('video_seg_{}'.format(process_id), (process_id%6)*300+60,int(process_id/6)*230)
                cv2.imshow('video_seg_{}'.format(process_id),img_)
                if cv2.waitKey(300) == 27:
                    flag_break =True
                    break
        else:
            break
    if vis:
        cv2.waitKey(30000)
        cv2.destroyWindow('video_seg_{}'.format(process_id))

def run_show(path,vis):
    seg_num = len(os.listdir(path))

    videos_path = os.listdir(path)
    # #--------------------------------------
    st_ = time.time()
    process_list = []
    for i in range(0,seg_num):
        # print(video_list[i])
        t = threading.Thread(target=run_one_process, args=(path + videos_path[i],i,vis))
        process_list.append(t)


    for i in range(0,seg_num):
        process_list[i].start()

    print(' start run ~ ')

    for i in range(0,seg_num):
        process_list[i].join()# 设置主线程等待子线程结束

    del process_list
    et_ = time.time()


if __name__ == "__main__":
    path = './video/'
    vis = True

    run_show_hights(path,vis)
    #--------------------------------------
