#encoding=utf-8
import sys
import argparse
import os
import cv2
import numpy as np

print(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..','deploy'))
from mtcnn_detector import MtcnnDetector
import mxnet as mx

class Detector(object):
    def __init__(self,args):
        self.args = args
        ctx = mx.gpu(args.gpu)
        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        mtcnn_path = os.path.join(os.path.dirname(__file__), '..','deploy', 'mtcnn-model')
        self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True, threshold=self.det_threshold)


    def get_face_landmarks(self,img):
        ret = self.detector.detect_face(img, det_type=self.args.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0,0:4]
        points = points[0,:].reshape((2,5)).T
        print(points)
        return points

from PIL import Image
import matplotlib.pyplot as plt
import imutils


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)),((nW / 2) - cX),((nH / 2) - cY)
    # return cv2.warpAffine(image, M, (nW, nH)),int(M[0, 2]), int(M[1, 2])


def main(args):
    face_path = '/media/handsome/data4/deepface_database/rgb_image/nir_real_deepface/img/78_0_3862_img.jpg'
    eye_glasses_path = "eye_glasses/0-15/87870318  1  -15  -180.png"
    draw_img = Image.open(face_path).convert('RGBA')
    eye_glasses = Image.open(eye_glasses_path)
    face_detector = Detector(args)
    img = cv2.imread(face_path)

    landmarks = face_detector.get_face_landmarks(img)
    print(landmarks)
    lec = [int(landmarks[0][0]), int(landmarks[0][1])]
    rec = [int(landmarks[1][0]), int(landmarks[1][1])]
    cv2.circle(img, (lec[0], lec[1]), 1, (255, 0, 0), 2)
    cv2.circle(img, (rec[0], rec[1]), 1, (0, 255, 0), 2)

    dY = rec[1] - lec[1]
    dX = rec[0] - lec[0]
    print("dydx",dY, dX)
    eye_dist = np.sqrt((dX ** 2) + (dY ** 2))
    eye_theta = np.arctan2(np.array([dY]), np.array([dX]))
    # eye_theta = np.arctan2(dY, dX)
    print("eye_theta",eye_theta)

    # plt.figure("eye_glasses")
    # plt.imshow(eye_glasses)
    # plt.show()
    # eye_glasses = cv2.imread("eye_glasses/0_30/1.png")
    # eye_glasses = cv2.cvtColor(eye_glasses,cv2.COLOR_BGR2BGRA)

    eye_glasses = np.asarray(eye_glasses)


    lower = np.array([254, 0, 0 , 255], dtype="uint8")
    upper = np.array([255, 0, 0, 255], dtype="uint8")
    mask = cv2.inRange(eye_glasses, lower, upper)
    output = cv2.bitwise_and(eye_glasses, eye_glasses, mask=mask)
    red_dot_index = np.where(output[:, :, 0:3] > 0)
    print("index:",red_dot_index)
    max_index = red_dot_index[1].argmax(axis=0)
    min_index = red_dot_index[1].argmin(axis=0)
    print("max",max_index)
    print("min", min_index)
    eye_glasses_pts = []
    eye_glasses_pts.append((red_dot_index[1][min_index],red_dot_index[0][min_index]))
    eye_glasses_pts.append((red_dot_index[1][max_index], red_dot_index[0][max_index]))
    print("eye_glasses_pts",eye_glasses_pts)

    import scipy
    pi = scipy.pi
    dot = scipy.dot
    sin = scipy.sin
    cos = scipy.cos
    ar = scipy.array
    # 围绕中心点旋转坐标
    def Rotate2D(pts, cnt, ang = pi / 4):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        return dot(pts - cnt, ar([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])) + cnt
    eye_cnt = np.array([eye_glasses.shape[1] // 2, eye_glasses.shape[0] // 2])
    rotate_point = [list(int(f) for f in Rotate2D(eye_glasses_pts[i], eye_cnt, ang= eye_theta[0])) for i in range(len(eye_glasses_pts))]

    eye_glasses,MX,MY = rotate_bound(eye_glasses, 180 * eye_theta[0] / np.pi)
    rotate_point[0][0] += MX
    rotate_point[0][1] += MY
    rotate_point[1][0] += MX
    rotate_point[1][1] += MY
    cv2.circle(eye_glasses, (rotate_point[0][0], rotate_point[0][1]), 1, (0, 255, 0), 2)
    cv2.circle(eye_glasses, (rotate_point[1][0], rotate_point[1][1]), 1, (0, 255, 0), 2)
    print("rect_point", rotate_point)

    egdx = rotate_point[1][0] - rotate_point[0][0]
    egdy = rotate_point[1][1] - rotate_point[0][1]
    eye_glasses_dist = np.sqrt((egdx ** 2) + (egdy ** 2))

    dist_ratio = eye_dist / eye_glasses_dist
    eye_glasses = cv2.resize(eye_glasses,(int(eye_glasses.shape[1] * dist_ratio), int(eye_glasses.shape[0] * dist_ratio)))
    paste_pts = [lec[0] - int(rotate_point[0][0]  * dist_ratio), lec[1] - int(rotate_point[0][1] * dist_ratio)]
    eye_glasses = cv2.GaussianBlur(eye_glasses,(3,3),0,0)
    eye_glasses = Image.fromarray(eye_glasses)
    draw_img.paste(eye_glasses, paste_pts, eye_glasses)

    draw_img = cv2.cvtColor(np.asarray(draw_img), cv2.COLOR_BGR2RGB)
    cv2.imshow("img",draw_img)
    cv2.waitKey(0)




def get_args():
    parser = argparse.ArgumentParser(description='face add eye glasses!!!')
    # general
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)