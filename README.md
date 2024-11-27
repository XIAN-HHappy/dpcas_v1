# dpcas_v1
### dpcas(Deep Learning Componentized Application System)：深度学习组件化应用系统。
* 为了更好更快的将已有的模型进行快速集成，实现应用。
* 该项目会尝试推出其它类型或是不同行业领域的项目应用。

## 项目简介
### [1、DpCas架构介绍](#DpCas架构)
### [2、算法组件](#算法组件)
### [3、项目案例](#项目案例)
## DpCas架构介绍
### DpCas FrameWork
* DpCas的FrameWork如下图所示：
![dpcas_framework](https://codechina.csdn.net/EricLee/dpcas/-/raw/master/DpCasFrameWork.png)
### xxx_lib（应用支持库）
* 具体应用对应的支持库，包括3部分如下图所示：
![dpcas_framework](https://codechina.csdn.net/EricLee/dpcas/-/raw/master/xxx_lib.png)
### Application（应用 pipeline）
* 具体应用流程，则是基于项目业务设计的pipeline的具体实现，以业务流为指导，去调用应用支持库（xxx_lib）和 算法组件（components）。

## 算法组件
- [x] [手部检测（yolo_v3）](https://github.com/XIAN-HHappy/yolo_v3)  
- [x] [人脸检测（yolo_v3）](https://github.com/XIAN-HHappy/yolo_v3)  
- [x] [人检测（yolo_v3）](https://github.com/XIAN-HHappy/yolo_v3)  
- [x] [安全帽检测（yolo_v3）](https://github.com/XIAN-HHappy/yolo_v3)  
- [x] [交通工具检测（yolo_v3）](https://github.com/XIAN-HHappy/yolo_v3)  
- [x] [coco物体检测（yolo_v3）](https://github.com/XIAN-HHappy/yolo_v3)  
- [x] [手部21关键点检测（handpose_x）](https://github.com/XIAN-HHappy/handpose_x)  
- [x] [物体识别分类（classification）](https://codechina.csdn.net/EricLee/classification)  
- [x] [人脸关键点（facial_landmark）](https://codechina.csdn.net/EricLee/facial_landmark)  
- [x] [人脸识别（insight_face）](https://codechina.csdn.net/EricLee/insight_face)  
- [x] [人像分割（faceparsing）](https://codechina.csdn.net/EricLee/faceparsing)  
- [x] [人体关键点（light_pose）](https://codechina.csdn.net/EricLee/light_pose)  

## 项目案例
### 项目1：手势交互项目（local 本地版本）
* [手势交互项目-具体介绍](https://github.com/XIAN-HHappy/handpose_x/blob/main/README.md)  

 ![image_dogs](samples/handpose_x.png)  
* [Demo完整视频](https://www.bilibili.com/video/BV1tX4y137tG/)
### 项目2：手势识别项目（二维关键点约束方式）
 ![image_gesture](./samples/gesture2.png)
### 项目3：Who You Want To See "你想看谁"
* [Who You Want To See "你想看谁"-具体介绍](lib/wyw2s_lib/doc/README.md)  

 ![wyw2s](samples/wyw2s-a.png)  

 ![wyw2s-b](samples/wyw2s-b.png)   
* [Demo完整视频](https://www.bilibili.com/video/BV1Z54y1b7zU/)

### 项目4：Face Bioassay "基于人脸动作的活体检测"
* [Face Bioassay "基于人脸动作的活体检测"](lib/face_bioassay_lib/doc/README.md)  

 ![bioassay-a](samples/bioassay-a.png)  

 ![bioassay-b](samples/bioassay-b.png)   
* [Demo完整视频](https://www.bilibili.com/video/BV1CK4y1G7j7/)

## 联系方式 （Contact）  
* E-mails: 305141918@qq.com   
