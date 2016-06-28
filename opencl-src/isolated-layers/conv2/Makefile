
all:
	echo ------3D CPU----******
	echo
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dcpu | more
	echo ------3D GPU----*****
	echo
	gcc -DDEVICE_GPU -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dgpu | more

3dcpu:
	gcc -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dcpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dcpu | more

3dgpu:
	gcc -DDEVICE_GPU -I/usr/local/include 3Dhost.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv3Dgpu -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv3Dgpu | more

clean:
	rm -rf conv* output*
