# make facebank
import warnings
warnings.filterwarnings("ignore")
import os
import torch
from model import Backbone
import argparse
from pathlib import Path
from torchvision import transforms as trans
from PIL import Image
import numpy as np
def prepare_facebank(path_images,facebank_path, model, mtcnn, device , tta = True):
    #
    test_transform_ = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    #
    model.eval()
    embeddings =  []
    names = ['Unknown']
    idx = 0
    for path in path_images.iterdir():
        if path.is_file():
            continue
        else:
            idx += 1
            print("idx {} : {}".format(idx,path))
            embs = []
            for file in path.iterdir():
                # print(file)
                if not file.is_file():
                    continue
                else:

                    try:
                        # print("---------------------------")
                        img = Image.open(file)
                        print(" {}) {}".format(idx,file))
                    except:
                        continue

                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(test_transform_(img).to(device).unsqueeze(0))
                            emb_mirror = model(test_transform_(mirror).to(device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(test_transform_(img).to(device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, facebank_path+'/facebank.pth')
    np.save(facebank_path + '/names', names)
    return embeddings, names

if __name__ == '__main__':
    # 需要制作人脸库对应的 图片地址
    path_images = "./images/"


    # 定义模型
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ = Backbone(50, 1., "ir_se").to(device_)
    # 加载模型
    if os.access("./model_ir_se50.pth",os.F_OK):
        model_.load_state_dict(torch.load("./model_ir_se50.pth"))

    model_.eval()
    facebank_path = "./facebank/" # 人脸库对应地址
    targets, names = prepare_facebank(Path(path_images), facebank_path,model_, "" ,device_, tta = False) # 构建 人脸 底库
