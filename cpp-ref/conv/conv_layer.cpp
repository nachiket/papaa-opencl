/*
Author: Gopalakrishna Hegde, NTU Singapore
Date: 6 July 2016

Test case for simple 2D filter.

*/

#include <stdio.h>
#include "pgm.h"

typedef float IMG_TYPE;

typedef struct {
	int rows;
	int cols;
	IMG_TYPE *data;
}Mat;

// 2D filter. This is simplified version of cv::filter2D
void customFilter2D(const Mat &input, Mat &out, const Mat &ker, float bias) {
	
	for(int r = 0; r < input.rows - ker.rows + 1; r++) {
		for(int c = 0; c < input.cols - ker.cols + 1; c++) {
			out.data[r*input.rows+c] = 0;
			for(int i = 0; i < ker.rows; i++) {
				for(int j = 0; j < ker.cols; j++) {
					out.data[r*input.rows+c] += ker.data[i*ker.cols+j] * input.data[(r+i)*input.cols + c +j];
				}
			}
			out.data[r*input.rows+c] += bias;
		}
	}
}



void print_help(char **argv) {
	printf("Usage : %s <image path>\n", argv[0]);
}

int main(int argc, char **argv) {

	char * imgName = NULL;
	pgm_t inputImg, filtImg;

	if(argc == 1) {
		print_help(argv);
		return -1;
	}
	imgName = argv[1];

	readPGM(&inputImg, imgName);
    // Input normalization to make it in the range [0, 1], assuming the input image to be in the range [0, 255]
    Mat normInput;
	normInput.data = (IMG_TYPE*)malloc(inputImg.width*inputImg.height*sizeof(IMG_TYPE));
	normInput.rows = inputImg.height;
	normInput.cols = inputImg.width;

	for(int p = 0; p < inputImg.width*inputImg.height; p++) {
    	normInput.data[p] = (IMG_TYPE)inputImg.buf[p]/255.0;
	}

	// create a sample Guassian kernel
    Mat kernel, output;
	const int kernelSize = 3;
	IMG_TYPE lap_ker[kernelSize*kernelSize] = {-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0};
	kernel.data = lap_ker;
	kernel.rows = kernelSize;
	kernel.cols = kernelSize;
	
	output.data = (IMG_TYPE*)malloc(inputImg.width*inputImg.height*sizeof(IMG_TYPE));
	output.rows = inputImg.height;
	output.cols = inputImg.width;

	// perform filtering
	float bias = 0.01;
	customFilter2D(normInput, output, kernel, bias);
	filtImg.width = output.cols;
	filtImg.height = output.rows;

	normalizeF2PGM(&filtImg, output.data);

	writePGM(&filtImg, "output.pgm");
	return 0;
}
