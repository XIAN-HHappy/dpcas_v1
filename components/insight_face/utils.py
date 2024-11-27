from datetime import datetime
from PIL import Image
import numpy as np
import io
from torchvision import transforms as trans

import torch
from insight_face.model import l2_norm
import pdb
import cv2

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

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
            print()
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                        print(" {}) {}".format(idx+1,file))
                    except:
                        continue
                    if img.size != (112, 112):
                        try:
                            img = mtcnn.align(img)
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

def load_facebank(facebank_path):
    embeddings = torch.load(facebank_path + '/facebank.pth')
    names = np.load(facebank_path + '/names.npy')
    return embeddings, names

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame


def infer(model, device, faces, target_embs, threshold = 1.2 ,tta=False):
    '''
    faces : list of PIL Image
    target_embs : [n, 512] computed embeddings of faces in facebank
    names : recorded names of faces in facebank
    tta : test time augmentation (hfilp, that's all)
    '''
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    #
    embs = []

    for img in faces:
        if tta:
            mirror = trans.functional.hflip(img)
            emb = model(test_transform(img).to(device).unsqueeze(0))
            emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
        else:
            with torch.no_grad():
                embs.append(model(test_transform(img).to(device).unsqueeze(0)))
    source_embs = torch.cat(embs)

    diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)

    minimum, min_idx = torch.min(dist, dim=1)
    min_idx[minimum > threshold] = -1 # if no match, set idx to -1

    return min_idx, minimum
