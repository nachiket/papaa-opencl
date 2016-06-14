import sys, os
import cv2
import matplotlib.pyplot as plt
import numpy as np

assert ('CAFFE_ROOT' in os.environ) , 'Please set CAFFE_ROOT in the environment variable to point to the caffe installation directory'

caffe_pkg_path = os.path.join(os.environ['CAFFE_ROOT'], 'python')
sys.path.insert(0, caffe_pkg_path)
import caffe

if(len(sys.argv) < 2):
    print('Please specify one image file as input')
    sys.exit()

img_path = sys.argv[1]

assert(os.path.exists(img_path)), 'Invalid file path'

# set caffe mode to GPU
#caffe.set_mode_gpu()
#caffe.set_device(0)

# set CPU mode
caffe.set_mode_cpu()

# initialize caffe object with  the network definition file
net_def = '../nets/conv_layer.prototxt'

net = caffe.Net(net_def, caffe.TEST)

print net.blobs['data'].data.shape
print net.blobs['conv'].data.shape

# load the image
image = cv2.imread(img_path)

if(image.ndim == 2):   # gray scale image
    input_data = image[np.newaxis, np.newaxis, :, :]
else:                  # BGR image. OpenCV color image representation is BGR format with #of channels being 3rd dimension
    # exchange the channel dimension to be first dimension
    input_data = image.transpose((2, 0, 1))
    # add one more dimension representing batch size = 1 to be compatible with caffe.blobs
    input_data = input_data[np.newaxis, :, :, :]

# adapt the network input data shape to image shape and assign the data to input blob
net.blobs['data'].reshape(*input_data.shape)
net.blobs['data'].data[...] = input_data

# compute the convolution output by forward pass
net.forward()

print net.blobs['conv'].data[0, 0]





