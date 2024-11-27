# 项目2：Who You Want To See "你想看谁"

## 项目案例介绍

*  开发语言：100% python代码。
*  场景：将视频中目标人物的相关视频进行裁剪。

## 项目配置  
### 1、软件
* Python 3.7  
* PyTorch >= 1.5.1  
* opencv-python
* moviepy
* shutil

## 相关项目
### 1、脸部检测项目（yolo_v3）
* 项目地址：https://codechina.csdn.net/EricLee/yolo_v3
* 另外同学们可以根据自己需求替换检测模型。
### 2、人脸识别项目(Insight_face)
* 项目地址：https://codechina.csdn.net/EricLee/insight_face

### 项目预训练模型 package
* [项目预训练模型 package 下载地址(百度网盘 Password: ofzq )](https://pan.baidu.com/s/18pQoo710IWJ9kN3k6tETyQ)
```
face_yolo_416-20210418.pt # 人脸检测模型

euler_angle-resnet_18_imgsize_256.pth # 人脸姿态角 pitch yaw roll 模型

face_multitask-resnet_50_imgsize-256-20210411.pth # 性别、年龄、关键点 模型

face_verify-model_ir_se-50.pth  # 人脸识别特征抽取模型

facebank/facebank.pth # 人脸匹配资源库特征向量

facebank/names.npy # 人脸匹配资源库 face id，示例中的face id为人名字
```

* 目前示例提供的人脸资源库的具体face id 如下：
```
['AngelinaJolie' 'AnneHathaway' 'BradPitt' 'JenniferAniston'
 'JohnnyDepp' 'JudeLaw' 'NicoleKidman' 'ScarlettJohansson' 'TomCruise']
```

## 项目使用方法  

### 1、下载项目预训练模型 package 。
### 2、构建人脸匹配资源库（项目中已经生成了示例匹配库，如果不需要建立自己的人脸资源库此步骤可以跳过），相关脚本 [make_facebank.py](https://codechina.csdn.net/EricLee/dpcas/-/blob/master/lib/wyw2s_lib/make_facebank_tools/make_facebank.py)
### 3、打开配置文件 lib/wyw2s_lib/cfg/[wyw2s.cfg](https://codechina.csdn.net/EricLee/dpcas/-/blob/master/lib/wyw2s_lib/cfg/wyw2s.cfg) 进行相关参数配置，以下配置为参考示例，同学可以根据自己的实际路径进行配置，示例配置参数如下，请仔细阅读。
```
YouWantToSee=BradPitt # 你需要裁剪的 face id ，示例为人名字，需要与facebank/names.npy 和 facebank/facebank.pth 信息匹配

detect_model_path=./wyw2s_models/face_yolo_416-20210418.pt # 人脸检测模型
detect_model_arch=yolo # 模型类型
detect_input_size = 416 # 模型的图片输入尺寸
yolo_anchor_scale=1. # anchor 的缩放系数，默认 1
detect_conf_thres=0.4 # 人脸检测置信度，高于该置信度进行输出
detect_nms_thres=0.45 # 检测的nms阈值

face_verify_backbone_path=./wyw2s_models/face_verify-model_ir_se-50.pth # 人脸识别特征抽取模型地址
facebank_path=./wyw2s_models/facebank  # 人脸资源库地址
face_verify_threshold=1.2 # 人脸匹配阈值设定，低于该设定阈值认为匹配成功

face_multitask_model_path=./wyw2s_models/face_multitask-resnet_50_imgsize-256-20210411.pth # 人脸多任务（性别、年龄、关键点）模型地址

face_euler_model_path=./wyw2s_models/euler_angle-resnet_18_imgsize_256.pth # 模型姿态角（航向角、俯仰角、翻滚角）回归模型地址
```

### 4、下载示例视频
* [示例视频 下载地址(百度网盘 Password: jaqh )](https://pan.baidu.com/s/1CSbfA1nHDhfCyt4_2NSRQg)
* 或是用同学自己的示例视频,但是要识别并裁切目标人物视频，必须与facebank人脸资源库对应，如果目标人物不在人脸资源库，会显示 "Unknown" 。

### 5、运行 "Who You Want To See" 项目
* 打开main.py，做如下相关参数设置：
```
APP_P = 1 # 选择不同项目 id
cfg_file = "./lib/wyw2s_lib/cfg/wyw2s.cfg" # 选择配置文件
main_wyw2s(video_path = "./video/f1.mp4",cfg_file = cfg_file)# 设置视频路径，加载 who you want 2 see  应用
```
* 根目录下运行命令： python main.py

## 联系方式 （Contact）  
* E-mails: 305141918@qq.com   
