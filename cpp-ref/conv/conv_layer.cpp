/*
Author: Gopalakrishna Hegde, NTU Singapore
Date: 6 July 2016

Test case for simple 2D filter.

*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;



// 2D filter. This is simplified version of cv::filter2D
void customFilter2D(const Mat &input, Mat &out, const Mat &ker) {
	
	for(int r = 0; r < input.rows - ker.rows + 1; r++) {
		for(int c = 0; c < input.cols - ker.cols + 1; c++) {
			out.at<float>(r,c) = 0;
			for(int i = 0; i < ker.rows; i++) {
				for(int j = 0; j < ker.cols; j++) {
					out.at<float>(r,c) += ker.at<float>(i, j) * input.at<float>(r+i, c+j);
				}
			}
		}
	}
}



void print_help(char **argv) {
	printf("Usage : %s <image path>\n", argv[0]);
}

int main(int argc, char **argv) {

	char * imgName = NULL;
	if(argc == 1) {
		print_help(argv);
		return -1;
	}
	imgName = argv[1];

    Mat input = imread(imgName, IMREAD_GRAYSCALE);

    // Input normalization to make it in the range [0, 1], assuming the input image to be in the range [0, 255]
    Mat normInput;
    input.convertTo(normInput, CV_32F);
    normInput = normInput/255;

	// create a sample Guassian kernel
    Mat kernel = Mat::ones(3, 3, CV_32F)/9.0;
	Mat output = Mat::zeros(input.rows-kernel.rows+1, input.cols-kernel.cols+1, CV_32F);

	// perform filtering
	customFilter2D(normInput, output, kernel);

	imshow("input image", normInput);
	imshow("output image", output);

	waitKey();
	return 0;
}
