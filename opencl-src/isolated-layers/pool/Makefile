
all:
	gcc -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool2Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool2Dcpu | more
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool3Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool3Dcpu | more
	gcc -DDEVICE_GPU -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool2Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool2Dgpu | more
	gcc -DDEVICE_GPU -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool3Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool3Dgpu | more

2dcpu:
	gcc -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool2Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool2Dcpu | more
3dcpu:
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool3Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool3Dcpu | more

2dgpu:
	gcc -DDEVICE_GPU -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool2Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool2Dgpu | more
3dgpu:
	gcc -DDEVICE_GPU -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o pool3Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./pool3Dgpu | more


clean:
	rm -rf pool* output*
