#Isolated opencl layers

This is for demo purpose only. It has individual layers in each folder. Each app will run on lenet5 configuration and produces output in output3dxx.pgm

You should run the app only in the foillowing order conv ==> pool ==> conv2 ==> pool2 ==> classify.

This has a normalization for each layer and this reduces accuracy a lot. so this is just for demo purpose to see how the output looks at each stage. use mnist folder if you want to classify the digit.

BUILD and RUN:

./make in each folder in the exact order

Some folder has 2-D kernel and 3-D in both GPU and CPU.. all runtimes will be printed when you ran make..
Purpose is to optimize the each layer and compare scenarios and teach the same.
 

