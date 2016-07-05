
all: conv2Dcpu conv3Dcpu conv2Dgpu conv3Dgpu

conv2Dcpu: 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -O3 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dcpu -lOpenCL -lpapi
	./conv2Dcpu | more

conv3Dcpu: 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -O3 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dcpu -lOpenCL -lpapi
	./conv3Dcpu | more

conv2dgpu: 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -DDEVICE_GPU -O3 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dgpu -lOpenCL -lpapi
	./conv2Dgpu | more

conv3dgpu: 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -DDEVICE_GPU -O3 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dgpu -lOpenCL -lpapi
	./conv3Dgpu | more

clean:
	rm -rf conv* output*
