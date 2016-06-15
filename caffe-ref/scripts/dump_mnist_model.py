import sys, os
import numpy as np
from utils import dump_conv_weights, dump_ip_weights 

def dump_model(net_def_file, model_file):

    c_file_header = '/* Lenet5 model  weights and biases*/\n#include "lenet5_model.h"\n\n'
    h_file_header = ('/*Header file for model  weights and biases*/\n'
        '#ifndef _LENET5_MODEL_H_\n#define _LENET5_MODEL_H_\n#include <stdio.h>\n\n')

    h_file = open('./gen/lenet5_model.h', 'w')
    c_file = open('./gen/lenet5_model.c', 'w')
    h_file.write(h_file_header)
    c_file.write(c_file_header)
    h_file.close()
    c_file.close()
    caffe.set_mode_cpu()    
    net = caffe.Net(net_def_file, model_file, caffe.TEST)
    dump_conv_weights(net, 'conv1','./gen/lenet5_model.c', './gen/lenet5_model.h')
    dump_conv_weights(net, 'conv2','./gen/lenet5_model.c', './gen/lenet5_model.h')
    dump_ip_weights(net, 'ip1', './gen/lenet5_model.c', './gen/lenet5_model.h')
    dump_ip_weights(net, 'ip2', './gen/lenet5_model.c', './gen/lenet5_model.h')

    h_file = open('./gen/lenet5_model.h', 'a')
    h_file.write('#endif // _LENET5_MODEL_H_')
    h_file.close()

if __name__=='__main__':
    assert ('CAFFE_ROOT' in os.environ) , ('Please set CAFFE_ROOT in the environment'
        'variable to point to the caffe installation directory')

    caffe_pkg_path = os.path.join(os.environ['CAFFE_ROOT'], 'python')
    sys.path.insert(0, caffe_pkg_path)
    import caffe

    if(not os.path.isdir('./gen')):
        os.mkdir('./gen')

    net_file = sys.argv[1]
    model_file = sys.argv[2]
    dump_model(net_file, model_file)

