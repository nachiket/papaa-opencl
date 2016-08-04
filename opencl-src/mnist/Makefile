all: mnistcpu mnistgpu mnistfpga	

mnistfpga:
	gcc -DDEVICE_ACC host_mnist_fpga.c load_kernel.c lenet5_model.c -g -o mnistfpga -lm -lOpenCL -O3 -lpapi

mnistcpu:
	gcc host_mnist.c load_kernel.c lenet5_model.c -g -o mnistcpu -lm -lOpenCL -O3 -lpapi

mnistgpu:
	gcc -DDEVICE_GPU host_mnist.c load_kernel.c lenet5_model.c  -g -o mnistgpu -lm -lOpenCL -O3 -lpapi

clean:
	rm -rf mnist* output*
