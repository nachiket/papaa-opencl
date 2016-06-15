all:host_layer.c
	gcc -I/usr/local/include host_layer.c load_kernel.c -o layers -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./layers

clean:host_layer.c
	rm -rf layers
