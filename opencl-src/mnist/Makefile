all:
	gcc -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c -g -o mnistcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./mnistcpu | more
	gcc -DDEVICE_GPU -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c -o mnistgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./mnistgpu | more
cpu:
	gcc -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c -g -o mnistcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
#	./mnistcpu | more
gpu:
	gcc -DDEVICE_GPU -I/usr/local/include host_mnist.c load_kernel.c lenet5_model.c -o mnistgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./mnistgpu | more
clean:
	rm -rf mnist* output*
