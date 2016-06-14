import sys, os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, exposure

assert ('CAFFE_ROOT' in os.environ) , ('Please set CAFFE_ROOT in the environment' 
    'variable to point to the caffe installation directory')

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
net_def = '../nets/inner_prod_layer.prototxt'

net = caffe.Net(net_def, caffe.TEST)


# load the image
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
input_data = image[np.newaxis, np.newaxis, :, :]

# adapt the network input data shape to image shape and assign the data to input blob
net.blobs['data'].reshape(*input_data.shape)
net.blobs['data'].data[...] = input_data

# compute the convolution output by forward pass
net.forward()

print('Output of inner product layer')
print net.blobs['ip'].data[0]





