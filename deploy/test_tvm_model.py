import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd
import cv2
ctx = tvm.cpu()
# load the module back.
loaded_json = open("./deploy_graph.json").read()
loaded_lib = tvm.module.load("./deploy_lib.so")
loaded_params = bytearray(open("./deploy_param.params", "rb").read())
image_size = (112, 112)
opt_level = 3

shape_dict = {'data': (1, 3, *image_size)}
img1 = cv2.imread('Ricardo_Sanchez_0002.jpg')
img2 = cv2.imread('Richard_Crenna_0001.jpg')

# img1 = img1[...,::-1]
# img2 = img2[...,::-1]


img1 = np.reshape(img1,shape_dict['data'])
img2 = np.reshape(img2,shape_dict['data'])
input_data1 = tvm.nd.array(img1.astype("float32"))
input_data2 = tvm.nd.array(img2.astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)

# Tiny benchmark test.
import time
for i in range(100):
   t0 = time.time()
   module.run(data=input_data1)
   f1 = module.get_output(0).asnumpy()[0]

   module.run(data=input_data2)
   f2 = module.get_output(0).asnumpy()[0]
   print(time.time() - t0)

   num = np.dot(f1, f2)
   denom = np.linalg.norm(f1) * np.linalg.norm(f2)
   cosine = num / denom
   print(cosine)