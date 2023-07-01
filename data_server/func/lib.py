import glob
import torch 
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from func.face_comparison.facenet import loadModel
from func.face_comparison.compare_faces import img_to_encoding
import cv2
from facenet_pytorch import MTCNN
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import datetime
import json