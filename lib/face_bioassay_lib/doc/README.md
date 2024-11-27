# 项目3：Face Bioassay "基于人脸动作的活体检测"

## 项目案例介绍

*  开发语言：100% python代码。
*  场景：基于人脸动作的活体检测。

## 项目配置  
### 1、软件
* Python 3.7  
* PyTorch >= 1.5.1  
* opencv-python

## 相关项目
### 1、脸部检测项目（yolo_v3）
* 项目地址：https://codechina.csdn.net/EricLee/yolo_v3
* 另外同学们可以根据自己需求替换检测模型。

### 项目预训练模型 package
* [项目预训练模型 package 下载地址(百度网盘 Password: 4y67 )](https://pan.baidu.com/s/1X8JCovTVnB7Zwf91TxwPJg)
```
face_yolo_416-20210418.pt # 人脸检测模型

euler_angle-resnet_18_imgsize_256.pth # 人脸姿态角 pitch yaw roll 模型

face_multitask-resnet_50_imgsize-256-20210411.pth # 性别、年龄、关键点 模型

face_au-resnet50-size256-20210427.pth # 眼睛、嘴巴开闭状态 模型
```

## 项目使用方法  

### 1、下载项目预训练模型 package 。
### 2、打开配置文件 lib/face_bioassay_lib/cfg/[face_bioassay.cfg](https://codechina.csdn.net/EricLee/dpcas/-/blob/master/lib/face_bioassay_lib/cfg/face_bioassay.cfg) 进行相关参数配置，以下配置为参考示例，同学可以根据自己的实际路径进行配置，示例配置参数如下，请仔细阅读。
```
YouWantToSee=BradPitt # 你需要裁剪的 face id ，示例为人名字，需要与facebank/names.npy 和 facebank/facebank.pth 信息匹配

detect_model_path=./wyw2s_models/face_yolo_416-20210418.pt # 人脸检测模型
detect_model_arch=yolo # 模型类型
detect_input_size = 416 # 模型的图片输入尺寸
yolo_anchor_scale=1. # anchor 的缩放系数，默认 1
detect_conf_thres=0.4 # 人脸检测置信度，高于该置信度进行输出
detect_nms_thres=0.45 # 检测的nms阈值

face_multitask_model_path=./wyw2s_models/face_multitask-resnet_50_imgsize-256-20210411.pth # 人脸多任务（性别、年龄、关键点）模型地址

face_euler_model_path=./wyw2s_models/euler_angle-resnet_18_imgsize_256.pth # 模型姿态角（航向角、俯仰角、翻滚角）回归模型地址

face_au_model_path=./wyw2s_models/face_au-resnet50-size256-20210427.pth # 眼睛、嘴巴开闭状态
```

### 3、下载示例视频
* [示例视频 下载地址(百度网盘 Password: akt9 )](https://pan.baidu.com/s/1zjo5RDYgI97lIGM2c18RrQ)

### 4、运行 "Face Bioassay" 项目
* 打开main.py，做如下相关参数设置：
```
APP_P = 2 # 选择不同项目 id
cfg_file = "./lib/face_bioassay_lib/cfg/face_bioassay.cfg" # 选择配置文件
main_face_bioassay(video_path = "./video/f1.mp4",cfg_file = cfg_file)# 设置视频路径，加载 face_bioassay  应用
```
* 根目录下运行命令： python main.py

## 联系方式 （Contact）  
* E-mails: 305141918@qq.com   
