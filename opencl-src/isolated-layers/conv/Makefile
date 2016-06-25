
all:
	gcc -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv2Dcpu | more
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dcpu | more
	gcc -DDEVICE_GPU -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv2Dgpu | more
	gcc -DDEVICE_GPU -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dgpu | more

2dcpu:
	gcc -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv2Dcpu | more
3dcpu:
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dcpu | more

2dgpu:
	gcc -DDEVICE_GPU -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv2Dgpu | more
3dgpu:
	gcc -DDEVICE_GPU -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dgpu | more


clean:
	rm -rf conv* output*
