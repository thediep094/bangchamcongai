import boto3
import numpy as np
from facenet_pytorch import MTCNN
import torch
import cv2
from pymongo import MongoClient
import torch 
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from func.face_comparison.facenet import loadModel
from func.face_comparison.compare_faces import img_to_encoding
import torch
import numpy as np
import boto3
from flask import Flask,jsonify
import json
import botocore
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
url = "mongodb+srv://vudinhtruongan:demo@test.6o5s3eu.mongodb.net/test"
client = MongoClient(url)
db = client['user_management1']
collection = db['users']
aws_access_key_id = 'AKIAUJUMLJ3YNYNZ5KU6'
aws_secret_access_key = '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'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
bucket_name = 'truongan912'
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))



@app.route("/convert_vector/<string:id>", methods=['GET', 'POST'])
def convert_vector(id):
    file_name = f'{id}.jpg'

    try:
        embeds = []
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        model = loadModel()
        image_data = response['Body'].read()
        image_data = np.frombuffer(image_data, np.uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
        face_img = np.array(mtcnn(image_data))
        face = np.transpose(face_img, (1, 2, 0))
        detect_embeds1 = img_to_encoding(face, model)
        detect_embeds1 = l2_normalize(detect_embeds1)
        detect_embeds_ = torch.tensor(detect_embeds1)
        embeds.append(detect_embeds_.unsqueeze(0))
        embedding = torch.cat(embeds).mean(0, keepdim=True)
        eb = embedding.flatten()
        list_embed = (eb.numpy()).tolist()
        data = {
            "face": list_embed
        }
        json_data = json.dumps(data)
        api_url = f"http://localhost:3000/api/user/admin/update/face/{id}"
        response = requests.put(api_url, json = data)

        if response.status_code == 200:
            return jsonify({'id': id, 'message': 'OK'})
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return jsonify({'id': id, 'error': 'File not found'})
        else:
            return jsonify({'id': id, 'error': 'Error retrieving file'})
  

if __name__ == '__main__':
    app.run(debug=True)