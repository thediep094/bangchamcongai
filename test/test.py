import boto3
import numpy as np
from facenet_pytorch import MTCNN
import torch
import cv2
import os
import shutil
from pymongo import MongoClient
import tensorflow as tf
from func.face_comparison.facenet import loadModel

url = "mongodb+srv://vudinhtruongan:demo@test.6o5s3eu.mongodb.net/test"
client = MongoClient(url)
db = client['user_management']
collection = db['users']
aws_access_key_id = 'AKIAUJUMLJ3YNYNZ5KU6'
aws_secret_access_key = '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'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
bucket_name = 'truongan912'
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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