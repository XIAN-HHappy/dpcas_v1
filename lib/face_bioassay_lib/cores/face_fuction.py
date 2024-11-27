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
# date:2021-04-17
# Author: Eric.Lee
# function: pipline

import os
import numpy as np
import cv2
import torch
from PIL import Image

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        #return (intersect / (sum_area - intersect))*1.0
        return (intersect / (S_rec1 + 1e-6))*1.0

def draw_landmarks(img,output,face_w,face_h,x0,y0,vis = False):
    img_width = img.shape[1]
    img_height = img.shape[0]
    dict_landmarks = {}
    eyes_center = []
    x_list = []
    y_list = []
    for i in range(int(output.shape[0]/2)):
        x = output[i*2+0]*float(face_w) + x0
        y = output[i*2+1]*float(face_h) + y0

        x_list.append(x)
        y_list.append(y)

        if 41>= i >=33:
            if 'left_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['left_eyebrow'] = []
            dict_landmarks['left_eyebrow'].append([int(x),int(y),(0,255,0)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,0),-1)
        elif 50>= i >=42:
            if 'right_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['right_eyebrow'] = []
            dict_landmarks['right_eyebrow'].append([int(x),int(y),(0,255,0)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,0),-1)
        elif 67>= i >=60:
            if 'left_eye' not in dict_landmarks.keys():
                dict_landmarks['left_eye'] = []
            dict_landmarks['left_eye'].append([int(x),int(y),(255,55,255)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)
        elif 75>= i >=68:
            if 'right_eye' not in dict_landmarks.keys():
                dict_landmarks['right_eye'] = []
            dict_landmarks['right_eye'].append([int(x),int(y),(255,55,255)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)
        elif 97>= i >=96:
            eyes_center.append((x,y))
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (0,0,255),-1)
        elif 54>= i >=51:
            if 'bridge_nose' not in dict_landmarks.keys():
                dict_landmarks['bridge_nose'] = []
            dict_landmarks['bridge_nose'].append([int(x),int(y),(0,170,255)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (0,170,255),-1)
        elif 32>= i >=0:
            if 'basin' not in dict_landmarks.keys():
                dict_landmarks['basin'] = []
            dict_landmarks['basin'].append([int(x),int(y),(255,30,30)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (255,30,30),-1)
        elif 59>= i >=55:
            if 'wing_nose' not in dict_landmarks.keys():
                dict_landmarks['wing_nose'] = []
            dict_landmarks['wing_nose'].append([int(x),int(y),(0,255,255)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,255),-1)
        elif 87>= i >=76:
            if 'out_lip' not in dict_landmarks.keys():
                dict_landmarks['out_lip'] = []
            dict_landmarks['out_lip'].append([int(x),int(y),(255,255,0)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (255,255,0),-1)
        elif 95>= i >=88:
            if 'in_lip' not in dict_landmarks.keys():
                dict_landmarks['in_lip'] = []
            dict_landmarks['in_lip'].append([int(x),int(y),(50,220,255)])
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (50,220,255),-1)
        else:
            if vis:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)
    face_area = (max(x_list) - min(x_list))*(max(y_list) - min(y_list))
    return dict_landmarks,eyes_center,face_area

def draw_contour(image,dict,vis = False):
    x0 = 0# 偏置
    y0 = 0

    for key in dict.keys():
        # print(key)
        _,_,color = dict[key][0]

        if 'left_eye' == key:
            eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
            if vis:
                cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,255,55),-1)
        if 'right_eye' == key:
            eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
            if vis:
                cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,215,25),-1)

        if 'basin' == key or 'wing_nose' == key:
            pts = np.array([[dict[key][i][0]+x0,dict[key][i][1]+y0] for i in range(len(dict[key]))],np.int32)
            if vis:
                cv2.polylines(image,[pts],False,color,thickness = 2)

        else:
            points_array = np.zeros((1,len(dict[key]),2),dtype = np.int32)
            for i in range(len(dict[key])):
                x,y,_ = dict[key][i]
                points_array[0,i,0] = x+x0
                points_array[0,i,1] = y+y0

            # cv2.fillPoly(image, points_array, color)
            if vis:
                cv2.drawContours(image,points_array,-1,color,thickness=2)

def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 2)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [185, 195,190], thickness=tf, lineType=cv2.LINE_AA)
#-------------------------------------------------------------------------------

def face_alignment(imgn,eye_left_n,eye_right_n,\
desiredLeftEye=(0.34, 0.42),desiredFaceWidth=256, desiredFaceHeight=None):

    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    leftEyeCenter = eye_left_n
    rightEyeCenter = eye_right_n
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))


    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
	# determine the scale of the new resulting image by taking
	# the ratio of the distance between eyes in the *current*
	# image to the ratio of distance between eyes in the
	# *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,(leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    M_reg = np.zeros((3,3),dtype = np.float32)
    M_reg[0,:] = M[0,:]
    M_reg[1,:] = M[1,:]
    M_reg[2,:] = (0,0,1.)
    # print(M_reg)
    M_I = np.linalg.inv(M_reg)#矩阵求逆，从而获得，目标图到原图的关系
    # print(M_I)
    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    # cv_resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA]

    output = cv2.warpAffine(imgn, M, (w, h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)#

    #---------------------------------------------------------------------------------------

    # ptx1 = int(eye_left_gt_n[0]*M[0][0] + eye_left_gt_n[1]*M[0][1] + M[0][2])
    # pty1 = int(eye_left_gt_n[0]*M[1][0] + eye_left_gt_n[1]*M[1][1] + M[1][2])
    #
    # ptx2 = int(eye_right_gt_n[0]*M[0][0] + eye_right_gt_n[1]*M[0][1] + M[0][2])
    # pty2 = int(eye_right_gt_n[0]*M[1][0] + eye_right_gt_n[1]*M[1][1] + M[1][2])

    return output

def refine_face_bbox(bbox,img_shape):
    height,width,_ = img_shape

    x1,y1,x2,y2 = bbox

    expand_w = (x2-x1)
    expand_h = (y2-y1)

    x1 -= expand_w*0.12
    y1 -= expand_h*0.12
    x2 += expand_w*0.12
    y2 += expand_h*0.08

    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

    x1 = np.clip(x1,0,width-1)
    y1 = np.clip(y1,0,height-1)
    x2 = np.clip(x2,0,width-1)
    y2 = np.clip(y2,0,height-1)

    return (x1,y1,x2,y2)

def get_faces_batch_attribute(face_multitask_model,face_euler_model,face_au_model,dets,img_raw,use_cuda,face_size = 256,vis = False):
    if len(dets) == 0:
        return None
    img_align = img_raw.copy()
    # 绘制图像
    image_batch = None
    r_bboxes = []
    imgs_crop = []
    for b in dets:
        b = list(map(int, b))

        r_bbox = refine_face_bbox((b[0],b[1],b[2],b[3]),img_raw.shape)
        r_bboxes.append(r_bbox)
        img_crop = img_raw[r_bbox[1]:r_bbox[3],r_bbox[0]:r_bbox[2]]
        imgs_crop.append(img_crop)
        img_ = cv2.resize(img_crop, (face_size,face_size), interpolation = cv2.INTER_LINEAR) # INTER_LINEAR INTER_CUBIC

        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.

        img_ = img_.transpose(2, 0, 1)
        img_ = np.expand_dims(img_,0)

        if image_batch is None:
            image_batch = img_
        else:
            image_batch = np.concatenate((image_batch,img_),axis=0)

    landmarks_pre,gender_pre,age_pre = face_multitask_model.predict(image_batch)
    euler_angles = face_euler_model.predict(image_batch)

    faces_message = None
    faces_align_batch = None
    for i in range(len(dets)):
        x0,y0 = r_bboxes[i][0],r_bboxes[i][1]
        face_w = r_bboxes[i][2]-r_bboxes[i][0]
        face_h = r_bboxes[i][3]-r_bboxes[i][1]
        dict_landmarks,eyes_center,face_area = draw_landmarks(img_raw,landmarks_pre[i],face_w,face_h,x0,y0,vis = False)

        gray_ = cv2.cvtColor(img_align[r_bboxes[i][1]:r_bboxes[i][3],r_bboxes[i][0]:r_bboxes[i][2],:], cv2.COLOR_BGR2GRAY)

        blur_ = cv2.Laplacian(gray_, cv2.CV_64F).var()

        gender_max_index = np.argmax(gender_pre[i])#概率最大类别索引
        score_gender = gender_pre[i][gender_max_index]# 最大概率

        yaw,pitch,roll = euler_angles[i] # 欧拉角
        if vis:
            cv2.putText(img_raw, "yaw:{:.1f},pitch:{:.1f},roll:{:.1f}".format(yaw,pitch,roll),(int(r_bboxes[i][0]-1),int(r_bboxes[i][1]+36)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (253,139,54), 5)
            cv2.putText(img_raw, "yaw:{:.1f},pitch:{:.1f},roll:{:.1f}".format(yaw,pitch,roll),(int(r_bboxes[i][0]-1),int(r_bboxes[i][1]+36)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)

            cv2.putText(img_raw, "{}".format(int(face_area)),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]-3)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (253,39,54), 5) # face_area
            cv2.putText(img_raw, "{}".format(int(face_area)),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]-3)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)
        if gender_max_index == 1.:
            gender_str = "male"
        else:
            gender_str = "female"

        if abs(yaw)<45.:
            face_align_output = face_alignment(img_align,eyes_center[0],eyes_center[1],
                desiredLeftEye=(0.365, 0.38),desiredFaceWidth=256, desiredFaceHeight=None)
        else:
            face_align_output = face_alignment(img_align,eyes_center[0],eyes_center[1],
                desiredLeftEye=(0.38, 0.40),desiredFaceWidth=256, desiredFaceHeight=None)
        # au 输入 预处理
        face_align_output_ = cv2.cvtColor(face_align_output, cv2.COLOR_BGR2RGB).astype(np.float32)
        face_align_output_ = face_align_output_ - [123.67, 116.28, 103.53]
        face_align_output_ = face_align_output_.transpose(2, 0, 1)
        face_align_output_ = np.expand_dims(face_align_output_,0)
        if faces_align_batch is None:
            faces_align_batch = face_align_output_
        else:
            faces_align_batch = np.concatenate((faces_align_batch,face_align_output_),axis=0)


        #
        if faces_message is None:
            faces_message = []
        faces_message.append(
            {
                "xyxy":r_bboxes[i],
                "age":age_pre[i][0],
                "gender":gender_str,
                "head_3d_angle":{"yaw":yaw,"pitch":pitch,"roll":roll},
                "open_mouth":None,
                "eye_close":None
            }
            )
        # plot_box(r_bboxes[i][0:4], img_raw,label="{}, age: {:.1f}, unblur:{}".format(gender_str,age_pre[i][0],int(blur_)), color=(255,90,90), line_thickness=2)
        if vis:
            plot_box(r_bboxes[i][0:4], img_raw,label="{}, age: {:.1f}".format(gender_str,age_pre[i][0]), color=(255,90,90), line_thickness=2)
    # print("faces_align_batch shape : ",faces_align_batch.shape)
    au_features = face_au_model.predict(faces_align_batch)

    for i in range(len(dets)):
        # print("au_features : ",au_features[i])
        open_mouth = np.clip(au_features[i][8],0.,1.)
        close_eye_r = np.clip(au_features[i][0],0.,1.)
        close_eye_l = np.clip(au_features[i][1],0.,1.)
        print("open_mouth : {:.2f},eye_close: {:.3f} , {:.3f}".format(open_mouth,close_eye_r,close_eye_l))
        faces_message[i]["open_mouth"] = open_mouth
        faces_message[i]["eye_close"] = (close_eye_r,close_eye_l)
        # 根据阈值 判断 其 开闭状态
        r_eye_str = "close" if close_eye_r>0.35 else "open"
        l_eye_str = "close" if close_eye_l>0.35 else "open"
        open_mouth_str = "open" if open_mouth>0.26 else "close"
        # 开闭状态 可视化
        if open_mouth_str == "open":
            cv2.putText(img_raw, "{:.2f}:mouth {}".format(open_mouth,open_mouth_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+15)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (225,15,25), 3)
            cv2.putText(img_raw, "{:.2f}:mouth {}".format(open_mouth,open_mouth_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+15)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)
        else:
            cv2.putText(img_raw, "{:.2f}:mouth {}".format(open_mouth,open_mouth_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+15)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,25), 3)
            cv2.putText(img_raw, "{:.2f}:mouth {}".format(open_mouth,open_mouth_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+15)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)


        if r_eye_str == "open":
            cv2.putText(img_raw, "{:.2f}:r_eye {}".format(close_eye_r,r_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+45)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,25), 3)
            cv2.putText(img_raw, "{:.2f}:r_eye {}".format(close_eye_r,r_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+45)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)
        else:
            cv2.putText(img_raw, "{:.2f}:r_eye {}".format(close_eye_r,r_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+45)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (225,15,25), 3)
            cv2.putText(img_raw, "{:.2f}:r_eye {}".format(close_eye_r,r_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+45)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)
        if l_eye_str == "open":
            cv2.putText(img_raw, "{:.2f}:l_eye {}".format(close_eye_l,l_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+70)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,25), 3)
            cv2.putText(img_raw, "{:.2f}:l_eye {}".format(close_eye_l,l_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+70)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)
        else:
            cv2.putText(img_raw, "{:.2f}:l_eye {}".format(close_eye_l,l_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+60)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (225,15,25), 3)
            cv2.putText(img_raw, "{:.2f}:l_eye {}".format(close_eye_l,l_eye_str),(int(r_bboxes[i][0]-1),int(r_bboxes[i][3]+60)),cv2.FONT_HERSHEY_DUPLEX, 0.65, (20,185,255), 1)

    return faces_message
