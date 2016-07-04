from pylearn2.datasets.mnist import MNIST
import os, sys
import cv2
import numpy as np

test_imgs = MNIST(which_set='test', center=False)
test_imgs.X = np.reshape(test_imgs.X, (-1, 1, 28, 28))
dst_dir = os.path.join(os.getcwd(), 'mnist-testset')

if(not os.path.isdir(dst_dir)):
    os.mkdir(dst_dir)

print ('No of test images = {:d}'.format(test_imgs.X.shape[0]))
print type(test_imgs.y[0,0])
for im in range(test_imgs.X.shape[0]):
    img_name = os.path.join(dst_dir, 'mnist_test_img_'+str(im)+'.pgm')
    print('Writing {:s}'.format(img_name))
    cv2.imwrite(img_name, 255*test_imgs.X[im, 0, :, :])

with open(os.path.join(os.getcwd(), 'mnist_test_img_list.csv'), 'w') as lf:
    for im in range(test_imgs.y.shape[0]):
        lf.write('mnist_test_img_{:s}.pgm, {:d}\n'.format(str(im), int(test_imgs.y[im, :])))

