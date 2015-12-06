# file:   Makefile
# author: Matthew Triche
# brief:  Makefile for fbn

OUTPUT=fbn
CC=g++
LIB_PATH=/usr/local/lib
INC_PATH=/usr/local/include

all: fbn.cpp proc_kernel.o
	${CC} -o ${OUTPUT} -L${LIB_PATH} -I${INC_PATH} proc_kernel.o fbn.cpp -lopencv_core -lopencv_features2d -lopencv_gpu -lopencv_imgproc -lopencv_nonfree -lopencv_contrib -lopencv_highgui -lopencv_calib3d -lopencv_flann

proc_kernel: proc_kernel.h proc_kernel.cpp
	${CC} -c proc_kernel.cpp

clean:
	rm ./*.o

