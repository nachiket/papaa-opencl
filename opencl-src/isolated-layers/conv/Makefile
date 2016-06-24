
all:
	gcc -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2D -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv2D | more
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3D -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3D | more
2d:
	gcc -I/usr/local/include 2Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv2D -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv2D | more
3d:
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3D -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3D | more

clean:
	rm -rf conv* output*
