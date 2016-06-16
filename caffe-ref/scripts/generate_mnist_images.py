from pylearn2.datasets.mnist import MNIST
import cv2
import numpy as np

test_imgs = MNIST(which_set='test', center=False)
test_imgs.X = np.reshape(test_imgs.X, (-1, 1, 28, 28))

print test_imgs.X.shape 
for im in range(10):
    img_name = 'mnist_test_img_'+str(im)+'.pgm'
    print('Writing {:s}'.format(img_name))
    cv2.imwrite(img_name, 255*test_imgs.X[im, 0, :, :])

