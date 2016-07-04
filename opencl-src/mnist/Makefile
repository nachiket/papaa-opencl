all:
	gcc -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c -g -o mnistcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	gcc -DDEVICE_GPU -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c -g -o mnistgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
cpu:
	gcc -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c -g -o mnistcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
gpu:
	gcc -DDEVICE_GPU -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c  -g -o mnistgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
clean:
	rm -rf mnist* output*
