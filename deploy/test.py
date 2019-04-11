import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/media/handsome/backupdata1/hanson/face-recognition/models/mobilenet-face/model-y1-softmax12s,0', help='path to load model.')
parser.add_argument('--ga-model', default='../gender-age/model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread('01025442ulctzgy4.jpg')
cv2.imshow('1', img)
img = model.get_input(img)
f1 = model.get_feature(img)
#print(f1[0:10])
gender, age = model.get_ga(img)
print(gender)
print(age)
# sys.exit(0)
img = cv2.imread('01031553l701ngwg.jpg')
cv2.imshow('2', img)
img = model.get_input(img)
f2 = model.get_feature(img)
gender, age = model.get_ga(img)
print(gender)
print(age)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
cv2.waitKey(0)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
