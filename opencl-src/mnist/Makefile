all: mnistcpu mnistgpu mnistfpga	

mnistfpga:
	gcc -DDEVICE_ACC -I${XILINX_SDACCEL}/runtime/inc host_mnist_fpga.c load_kernel.c lenet5_model.c -g -o mnistfpga -lm -L${XILINX_SDACCEL}/runtime/lib/x86_64 -lxilinxopencl -O3 -lpapi

runfpga:
	./mnistfpga -x mnist/impl/sim/cpu_em/bin_mnist.xclbin -m sample -i lena.pgm

mnistcpu:
	gcc host_mnist.c load_kernel.c lenet5_model.c -g -o mnistcpu -lm -lOpenCL -O3 -lpapi

mnistgpu:
	gcc -DDEVICE_GPU host_mnist.c load_kernel.c lenet5_model.c  -g -o mnistgpu -lm -lOpenCL -O3 -lpapi

clean:
	rm -rf mnist* output*
