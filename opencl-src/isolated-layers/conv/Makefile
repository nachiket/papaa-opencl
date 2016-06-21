all:host_layer.c
	gcc -I/usr/local/include host_layer.c load_kernel.c conv_layer_weight.c -o conv -lm -lOpenCL -O3 /usr/local/lib/libpapi.a
	./conv | more

clean:host_layer.c
	rm -rf conv
