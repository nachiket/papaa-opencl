
all: compile run

compile: class_cpu class_gpu

class_cpu: class_host.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -O3 class_host.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dcpu -lOpenCL -lpapi

class_gpu: class_host.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c
	gcc -DDEVICE_GPU -O3 class_host.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dgpu -lOpenCL -lpapi

run:
	./class_cpu pool.pgm
	./class_gpu pool.pgm

clean:
	rm -rf class* output*

