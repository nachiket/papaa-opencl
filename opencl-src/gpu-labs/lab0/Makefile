include sources.mk
CC=gcc
CFLAGS= -g -O3
LIBS= OpenCL
INC_DIRS= /usr/local/cuda-8.0/include
LIB_DIRS= /usr/local/cuda-8.0/lib64

exec=lab0_app

all:$(exec)

$(exec):$(C_SRCS)
	@echo Building....
	$(CC) $(CFLAGS) $(C_SRCS) -o $(exec) \
	$(foreach D,$(INC_DIRS),-I$D) $(foreach D,$(LIB_DIRS),-L$D) \
	$(foreach L,$(LIBS),-l$L)

run:$(exec)
	./$(exec) ../../../imgs/mnist_test_img_0.pgm

pyrun:$(exec)
	python host_app_gpu.py ../../../imgs/mnist_test_img_0.pgm

print:
	@echo "C_SRCS = $(C_SRCS)"
	@echo "C_FLAGS = $(CFLAGS)"

.phony:clean
clean:
	rm -rf $(exec)
	rm -rf *.pgm
