import sys, os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, exposure

c_file_header = '/* Example convolution layer weights and biases*/\n#include "conv_layer_weight.h"\n\n'
h_file_header = ('/*Header file for convolution layer weights and biases*/\n'
    '#ifndef _CONV_LAYER_WEIGHT_H_\n#define _CONV_LAYER_WEIGHT_H_\n#include <stdio.h>\n\n')

def write_conv_weights(net):
    if(not os.path.isdir('./gen')):
        os.mkdir('./gen')

    conv_weights = net.params['conv'][0].data
    conv_bias = net.params['conv'][1].data

    h_file = open('./gen/conv_layer_weight.h', 'w')
    c_file = open('./gen/conv_layer_weight.c', 'w')
    h_file.write(h_file_header)
    c_file.write(c_file_header)

    # write # defines related to conv layer params
    h_file.write('#define CONV1_NO_INPUTS  '+str(conv_weights.shape[1])+'\n\n')
    h_file.write('#define CONV1_NO_OUTPUTS  '+str(conv_weights.shape[0])+'\n\n')
    h_file.write('#define CONV1_FILTER_HEIGHT  '+str(conv_weights.shape[2])+'\n\n')
    h_file.write('#define CONV1_FILTER_WIDTH  '+str(conv_weights.shape[3])+'\n\n')

    # extern variable weight and bias array names
    h_file.write('extern ' + 'const ' + 'float ' + 'conv1_weights' + 
        '[CONV1_NO_OUTPUTS][CONV1_NO_INPUTS*FILTER_HEIGHT*FILTER_WIDTH];\n\n')
    h_file.write('extern ' + 'const ' + 'float ' + 'conv1_bias' + '[CONV1_NO_OUTPUTS];\n\n')
    h_file.write('#endif // _CONV_LAYER_WEIGHT_H_')

    # write weights to the C source file
    c_file.write('const float conv1_weights[CONV1_NO_OUTPUTS][CONV1_NO_INPUTS*CONV1_FILTER_HEIGHT*CONV1_FILTER_WIDTH] = {\n')
    for f in range(conv_weights.shape[0]):
        c_file.write('{')
        filt = conv_weights[f].reshape(-1).tolist()
        for i, e in enumerate(filt):
            if(i == len(filt)-1):
                c_file.write('{:f}'.format(e))
            else:
                c_file.write('{:f}, '.format(e))
        if(f == conv_weights.shape[0]-1):
            c_file.write('}\n')
        else:
            c_file.write('},\n')
    c_file.write('};\n\n')

    # write bias to same file
    c_file.write('const ' + 'float ' + 'conv1_bias' + '[CONV1_NO_OUTPUTS] = {\n')
    bias = conv_bias.tolist()
    for i, b in enumerate(bias):
        if(i == len(bias)-1):
            c_file.write('{:f}'.format(b))
        else:
            c_file.write('{:f}, '.format(b))
    c_file.write('};\n\n')
    

    h_file.close()
    c_file.close()
    

if __name__=='__main__':
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
    net_def = '../nets/conv_layer.prototxt'
    
    net = caffe.Net(net_def, caffe.TEST)
    
    
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
    
    print (net.params['conv'][0].data.shape)
    print net.params['conv'][1].data.shape
    # display all maps
    for n in range(net.blobs['conv'].data.shape[1]):
        feat_map = exposure.rescale_intensity(net.blobs['conv'].data[0, n], out_range='float')
        cv2.imshow('conv_maps', feat_map)
        cv2.waitKey()

    # write the weights and biases to a file
    print('The weights and biases of this layer are written in ./gen directory')
    write_conv_weights(net)
    # store the caffe model
    net.save('./gen/random_conv_model.caffemodel')





