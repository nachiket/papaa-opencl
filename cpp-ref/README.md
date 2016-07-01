# Lenet5 C++ Application

# Description
This is a sipmle implementation of Lenet5 Convolutional Neural Net that can be found [here](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt) The model is trained using Caffe and weights are stored in lenet5_model.cpp

# Usage
Execute following commands to run the app. The app will print the predicted digit and all 10 probabilities.

make all

./lenet_app <28x28 grayscale image path>

Ex: ./lenet_app ../imgs/mnist_test_img_0.jpg
