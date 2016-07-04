# Lenet5 OpenCL Application

This is the OpenCL based code for lenet5 app. All the kernels are in kernels.cl file.
This code can be compiled for GPU or CPU.
The app will print the predicted digit and all 10 probabilities.

#BUILD:

make all --> builds both cpu and gpu opencl bytecodes.

(or)

make cpu --> builds only cpu bytecode

make gpu --> builds only gpu bytecode

#Run:

./mnistcpu <28x28 grayscale image path> --> to run in cpu

(or)

./mnistgpu <28x28 grayscale image path>--> to run in gpu


Ex: ./mnistcpu ../imgs/mnist_test_img_0.jpg
