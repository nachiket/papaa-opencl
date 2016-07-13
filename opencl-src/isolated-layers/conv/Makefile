
all: compile run

compile: conv2Dcpu conv3Dcpu conv2Dgpu conv3Dgpu

conv2Dcpu: 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -O3 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dcpu -lOpenCL -lpapi

conv3Dcpu: 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -O3 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dcpu -lOpenCL -lpapi

conv2Dgpu: 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -g -DDEVICE_GPU -O3 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dgpu -lOpenCL -lpapi

conv3Dgpu: 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -g -DDEVICE_GPU -O3 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dgpu -lOpenCL -lpapi

run:
	./conv2Dcpu mnist_test_img_0.pgm
	./conv2Dgpu mnist_test_img_0.pgm
#	./conv3Dcpu mnist_test_img_0.pgm
#	./conv3Dgpu mnist_test_img_0.pgm

clean:
	rm -rf conv* output*
