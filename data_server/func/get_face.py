import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os
import numpy as np
import boto3
import shutil

aws_access_key_id = 'AKIAUJUMLJ3YNYNZ5KU6'
aws_secret_access_key = '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'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
bucket_name = 'truongan912'
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def get_face(id):
    IMG_PATH = './fe_be/data'
    count = 10
    USR_PATH = os.path.join(IMG_PATH, id)
    leap = 1    

    mtcnn = MTCNN(margin = 20, keep_all=False, select_largest = True, post_process=False, device = device)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    while cap.isOpened() and count:
        isSuccess, frame = cap.read()
        if mtcnn(frame) is not None and leap%2:
            path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+str(count)))
            face_img = mtcnn(frame, save_path = path)
            count-=1
        leap+=1
        cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break   
    cap.release()
    cv2.destroyAllWindows()
    return face_img


def get_face_upload(id):
    IMG_PATH = './data_server/data'
    USR_PATH = os.path.join(IMG_PATH, id)  
    count = 1

    mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)

    # Đọc ảnh từ folder cho trước
    image_paths = USR_PATH
    for image_path in os.listdir(image_paths):
        image = cv2.imread(os.path.join(image_paths, image_path))
        path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+str(count)))
        face_img = mtcnn(image, save_path=path)
        os.remove(os.path.join(image_paths, image_path))
    return face_img


def check_exist():
    response = s3.list_objects_v2(Bucket=bucket_name)
    parent_folder = 'data_server/data'
    key_list = []
    for obj in response['Contents']:
        key = obj['Key']
        key = key.replace(".jpg", "")
        key_list.append(key)

    items = os.listdir(parent_folder)
    
    # Lặp qua các thư mục con trong folder
    for item in items:
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            if item in key_list:
    
                continue
            
            # Key không trùng với tên thư mục con, xóa thư mục
            shutil.rmtree(item_path)

def get_face_aws(): 
    response = s3.list_objects_v2(Bucket=bucket_name)
    parent_folder = 'data_server/data'
    for obj in response['Contents']:
        key = obj['Key']
        key = key.replace(".jpg", "")
        if not os.path.exists(os.path.join(parent_folder, key)):
            key = key+".jpg"
            response = s3.get_object(Bucket=bucket_name, Key=key)
            image_data = response['Body'].read()
            image_data = np.frombuffer(image_data, np.uint8)
            image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
            face_img = mtcnn(image_data, save_path = f'./data_server/data/{key}.jpg') 
    image_files = [file for file in os.listdir(parent_folder) if file.lower().endswith('.jpg')]            
    for image_file in image_files:                
        image_name = os.path.splitext(image_file)[0]
        folder_name = image_name.replace(".jpg", "")
        folder_path = os.path.join(parent_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        source_path = os.path.join(parent_folder, image_file)
        destination_path = os.path.join(folder_path, image_file)
        shutil.move(source_path, destination_path)


