all:host_layer.c
	gcc -I/usr/local/include host_layer.c load_kernel.c ../../../caffe-ref/scripts/gen/lenet5_model.c -o conv -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv | more

clean:host_layer.c
	rm -rf conv output*
