# Lenet5 C++ Application

# Description
This is a sipmle implementation of Lenet5 Convolutional Neural Net that can be found
[here](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt) The model is trained using Caffe and weights are stored in
lenet5_model.cpp.

The app has 2 modes. One is for testing a sample image and the other is to test the full MNIST testset to determine the prediction accuracy.

# Usage
1. Build the app
make all
2. Execute the following command to run the app in 'sample' mode.
./lenet_app -m sample -i ../imgs/mnist_test_img_0.pgm
3. To run the app in the full 'test' mode, untar the ../imgs/mnist-testset.tar.gz and then execute 
./lenet_app -m test -f ../imgs/mnist_test_img_list.csv -d ../imgs/mnist-testset

