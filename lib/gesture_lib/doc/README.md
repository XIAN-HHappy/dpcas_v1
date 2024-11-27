# 项目2：手势识别项目（二维关键点约束方式）

## 项目案例介绍


## 项目配置  
### 1、软件  
* Python 3.7  
* PyTorch >= 1.5.1  
* opencv-python
* playsound
### 2、硬件
* 普通USB彩色（RGB）网络摄像头

## 相关项目
### 1、手部检测项目（yolo_v3）
* 项目地址：https://codechina.csdn.net/EricLee/yolo_v3
* [预训练模型下载地址(百度网盘 Password: 7mk0 )](https://pan.baidu.com/s/1hqzvz0MeFX0EdpWXUV6aFg)
* 另外同学们可以根据自己需求替换检测模型。
### 2、手21关键点回归项目(handpose_x)
* 项目地址：https://codechina.csdn.net/EricLee/handpose_x
* [预训练模型下载地址(百度网盘 Password: 99f3 )](https://pan.baidu.com/s/1Ur6Ikp31XGEuA3hQjYzwIw)

## 项目使用方法  
### 项目1：手势交互项目（local 本地版本）
### 1、下载手部检测模型和21关键点回归模型。
### 2、确定摄像头连接成功。
### 3、打开配置文件 lib/hand_lib/cfg/[handpose.cfg](https://codechina.csdn.net/EricLee/dpcas/-/blob/master/lib/hand_lib/cfg/handpose.cfg) 进行相关参数配置，具体配置参数如下，请仔细阅读（一般只需要配置模型路径及模型结构）
```
detect_model_path=./latest_416.pt #手部检测模型地址
detect_model_arch=yolo_v3 #检测模型类型 ，yolo  or yolo-tiny
yolo_anchor_scale=1.0 # yolo anchor 比例，默认为 1
detect_conf_thres=0.5 # 检测模型阈值
detect_nms_thres=0.45 # 检测模型 nms 阈值

handpose_x_model_path=./ReXNetV1-size-256-wingloss102-0.1063.pth # 21点手回归模型地址
handpose_x_model_arch=rexnetv1 # 回归模型结构

camera_id = 0 # 相机 ID ，一般默认为0，如果不是请自行确认
vis_gesture_lines = True # True: 点击时的轨迹可视化， False：点击时的轨迹不可视化
charge_cycle_step = 32 # 点击稳定状态计数器，点击稳定充电环。
```
### 4、根目录下运行命令： python main.py

## 联系方式 （Contact）  
* E-mails: 305141918@qq.com   
