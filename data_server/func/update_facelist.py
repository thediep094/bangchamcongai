import glob
import torch 
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from func.face_comparison.facenet import loadModel
from func.face_comparison.compare_faces import img_to_encoding
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import boto3
import tensorflow as tf


device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def trans(img):
    transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform(img).unsqueeze(0)



def update_facelist():
    IMG_PATH = './data_server/data'
    DATA_PATH = './data_server/path'

    model = loadModel()
    embeddings = []
    names = []

    for usr in os.listdir(IMG_PATH):
        embeds = []
        for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
            # print(usr)
            try:
                img = Image.open(file)
                
            except:
                continue
            with torch.no_grad():
                
                face = np.array(img)
                detect_embeds1 = img_to_encoding(face, model)
                detect_embeds1 = l2_normalize(detect_embeds1)
                detect_embeds_ = torch.tensor(detect_embeds1)
                embeds.append(detect_embeds_.unsqueeze(0))
        if len(embeds) == 0:
            continue
    
        embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
        embeddings.append(embedding) 
        names.append(usr)
    embeddings = torch.cat(embeddings) #[n,512]
    names = np.array(names)
    
    if device == 'cpu':
        torch.save(embeddings, DATA_PATH+"/faceslistCPU.pth")
    else:
        torch.save(embeddings, DATA_PATH+"/faceslist.pth")
    np.save(DATA_PATH+"/usernames", names)
    embedding_ = embeddings.numpy()
    print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))
    
    return embedding_, names

def update_face_vector():
    IMG_PATH = './data_server/data'


    model = loadModel()
    embeddings = []
    names = []

    for usr in os.listdir(IMG_PATH):
        embeds = []
        for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
            # print(usr)
            try:
                img = Image.open(file)
                
            except:
                continue
            with torch.no_grad():
                
                face = np.array(img)
                detect_embeds1 = img_to_encoding(face, model)
                detect_embeds1 = l2_normalize(detect_embeds1)
                detect_embeds_ = torch.tensor(detect_embeds1)
                embeds.append(detect_embeds_.unsqueeze(0))
        if len(embeds) == 0:
            continue
    
        embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
        embeddings.append(embedding) 
        names.append(usr)
    embeddings = torch.cat(embeddings) #[n,512]
    names = np.array(names)
    embedding_ = embeddings.numpy()
    return embedding_, names

