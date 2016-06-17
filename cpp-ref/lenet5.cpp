#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include "lenet5_model.h"

using namespace cv;
using namespace std;

typedef struct ConvWtBias {
	Mat W;
	Mat b;
 	int no_inputs;
	int no_outputs;
} ConvParams;

typedef struct FcWtBias {
	Mat W;
	Mat b;
	int no_inputs;
	int no_outputs;
} FcParams;

int main(int argc, char **argv) {

	if(argc < 2) {
		cout<<"Please specify the image path"<<endl;
		exit(0);
	}

	Mat input = imread(argv[1], IMREAD_GRAYSCALE);

	namedWindow("Input image", CV_WINDOW_AUTOSIZE);
	imshow("Input image", input);
    cout << input.channels() << endl;
	cout<<argv[1]<<endl;
	cout<<"Application complete"<<endl;
	waitKey(0);
	return 0;
}
