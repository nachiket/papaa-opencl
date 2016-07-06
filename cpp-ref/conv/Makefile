include ./sources.mk
exec=conv_layer

CC=g++
CPPFLAGS= -g $(shell pkg-config --cflags opencv) -I/usr/local/include/opencv2
LIBS=$(shell pkg-config --libs opencv)

OBJ_DIR= obj
CPP_OBJS := $(addprefix $(OBJ_DIR)/,$(notdir $(patsubst %.cpp,%.o,$(filter %.cpp,$(CPP_SRCS)))))

.phony: all
all: obj $(exec)

#vpath %.c $(sort $(dir $(C_SRCS)))

$(CPP_OBJS): $(OBJ_DIR)/%.o: %.cpp
	@echo Building $< ......
	@echo $(CPPFLAGS)
	$(CC) -c $(CPPFLAGS) $< -o $@

#executable
$(exec) : $(CPP_OBJS)
	@echo Linking.......
	$(CC)  -o $@ $(CPP_OBJS) $(CPPFLAGS) $(LIBS)

obj:
	@mkdir -p $(OBJ_DIR)
run: $(exec)
	./$(exec) ../caffe-ref/imgs/lena.jpg

print:
	@echo "CPP_SRCS = $(CPP_SRCS)"
	@echo "OBJS = $(CPP_OBJS)"
	@echo "CPPFLAGS = $(CPPFLAGS)"

.phony:clean
clean:
	rm -rf ./$(OBJ_DIR)
	rm $(exec)
