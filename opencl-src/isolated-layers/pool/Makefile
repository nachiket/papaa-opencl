
all: compile run

compile: pool2Dcpu pool2Dgpu pool3Dcpu pool3Dgpu

pool2Dcpu: 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -O3 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool2Dcpu -lOpenCL -lpapi

pool3Dcpu: 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -O3 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool3Dcpu -lOpenCL -lpapi

pool2Dgpu: 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -DDEVICE_GPU -O3 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool2Dcpu -lOpenCL -lpapi

pool3Dgpu: 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -DDEVICE_GPU -O3 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool3Dcpu -lOpenCL -lpapi

run:
	./pool2Dcpu mnist_test_img_0.pgm
	./pool2Dgpu mnist_test_img_0.pgm
	./pool3Dcpu mnist_test_img_0.pgm
	./pool3Dgpu mnist_test_img_0.pgm

clean:
	rm -rf pool* output*
