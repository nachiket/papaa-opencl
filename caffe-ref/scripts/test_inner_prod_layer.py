import sys, os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, exposure

c_file_header = '/* Example inner-product layer weights and biases*/\n#include "ip_layer_weight.h"\n\n'
h_file_header = ('/*Header file for inner-product layer weights and biases*/\n'
    '#ifndef _IP_LAYER_WEIGHT_H_\n#define _IP_LAYER_WEIGHT_H_\n#include <stdio.h>\n\n')

def write_ip_weights(net):
    if(not os.path.isdir('./gen')):
        os.mkdir('./gen')

    ip_weights = net.params['ip'][0].data
    ip_bias = net.params['ip'][1].data

    h_file = open('./gen/ip_layer_weight.h', 'w')
    c_file = open('./gen/ip_layer_weight.c', 'w')
    h_file.write(h_file_header)
    c_file.write(c_file_header)

    # write # defines related to conv layer params
    h_file.write('#define NO_INPUT  '+str(ip_weights.shape[1])+'\n\n')
    h_file.write('#define NO_OUTPUT  '+str(ip_weights.shape[0])+'\n\n')

    # extern variable weight and bias array names
    h_file.write('extern ' + 'const ' + 'float ' + 'ip_layer_weights' + 
        '[NO_OUTPUT][NO_INPUT];\n\n')
    h_file.write('extern ' + 'const ' + 'float ' + 'ip_layer_bias' + '[NO_OUTPUT];\n\n')
    h_file.write('#endif // _IP_LAYER_WEIGHT_H_')

    # write weights to the C source file
    c_file.write('const float ip_layer_weights[NO_OUTPUT][NO_INPUT] = {\n')
    for f in range(ip_weights.shape[0]):
        c_file.write('{')
        filt = ip_weights[f].tolist()
        for i, e in enumerate(filt):
            if(i == len(filt)-1):
                c_file.write('{:f}'.format(e))
            else:
                c_file.write('{:f}, '.format(e))
        if(f == ip_weights.shape[0]-1):
            c_file.write('}\n')
        else:
            c_file.write('},\n')
    c_file.write('};\n\n')

    # write bias to same file
    c_file.write('const ' + 'float ' + 'ip_layer_bias' + '[NO_OUTPUT] = {\n')
    bias = ip_bias.tolist()
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
    
    print net.params['ip'][0].data.shape
    print net.params['ip'][1].data.shape
    print('Output of inner product layer')
    print net.blobs['ip'].data[0]

    # write the parameters to C files
    write_ip_weights(net)





