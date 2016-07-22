include sources.mk
CC=gcc
CFLAGS= -g -O3
LIBS= -framework OpenCL 

exec=lab1_app

all:$(exec)

$(exec):$(C_SRCS)
	@echo Building....
	$(CC) $(CFLAGS) $(C_SRCS) -o $(exec) $(LIBS)

run:$(exec)
	./$(exec) ../../../imgs/mnist_test_img_0.pgm

print:
	@echo "C_SRCS = $(C_SRCS)"
	@echo "C_FLAGS = $(CFLAGS)"

.phony:clean
clean:
	rm -rf $(exec)
	rm -rf *.pgm
