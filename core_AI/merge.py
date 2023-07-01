import glob
import torch 
import cv2
import os
from facenet_pytorch import MTCNN
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from face_comparison.compare_faces import img_to_encoding
from face_comparison.facenet import loadModel
import datetime
import json
import time



frame_size = (640,480)
IMG_PATH = './MTCNN_Facenet512/data/test_images'
DATA_PATH = './MTCNN_Facenet512/data'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# def trans(img):
#     transform = transforms.Compose([
#             transforms.ToTensor(),
#             fixed_image_standardization
#         ])
#     return transform(img)

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def trans(img):
    transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform(img).unsqueeze(0)

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
            # print('smt')

            # embeds.append(loadModel(trans(img))) #1 anh, kich thuoc [1,512]
            
            face = np.array(img)
            detect_embeds1 = img_to_encoding(face, model)
            detect_embeds1 = l2_normalize(detect_embeds1)
            detect_embeds_ = torch.tensor(detect_embeds1)
            embeds.append(detect_embeds_.unsqueeze(0))

            
        
    if len(embeds) == 0:
        continue
    # embeds_ = torch.tensor(embeds)
    embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
    
    embeddings.append(embedding) # 1 cai list n cai [1,512]
    # print(embedding)
    names.append(usr)
    
embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)

if device == 'cpu':
    torch.save(embeddings, DATA_PATH+"/faceslistCPU.pth")
else:
    torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))


def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

def inference(model, face1, local_embeds, threshold = 3):
    
    face = np.array(face1)
    detect_embeds1 = img_to_encoding(face, model)
    detect_embeds1 = l2_normalize(detect_embeds1)
    detect_embeds = torch.tensor(detect_embeds1)
    
                    #[1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi

    min_dist, embed_idx = torch.min(norm_score, dim = 1)
   
    if min_dist > 0.4:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face


avalable_list = {}

if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    people_list = {}
    people_check = {}
    model = loadModel()
   

    mtcnn = MTCNN(thresholds= [0.6, 0.7, 0.7] ,keep_all=True, device = device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    embeddings, names = load_faceslist()
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = extract_face(bbox, frame)
                    idx, score = inference(model, face, embeddings)
                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy()
                        frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

                        people_list[str(datetime.datetime.now())]=str(names[idx])
                        print(names[idx] + '_{:.2f}'.format(score))
                        
                        
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

                    
                        
        

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)


    cap.release()
    cv2.destroyAllWindows()

    print(avalable_list)