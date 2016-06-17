#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include "lenet5_model.h"

using namespace cv;
using namespace std;

typedef struct ConvWtBias {
	vector<vector<Mat> >  W;
	vector<float> bias;
 	int noInputs;
	int noOutputs;
	int filtH;
	int filtW;
} ConvParams;

typedef struct FcWtBias {
	Mat W;
	vector<float> bias;
	int noInputs;
	int noOutputs;
} FcParams;

typedef struct ConvLayers {
	vector<ConvParams> layerParams;
	int noLayers;
} ConvLayers;

typedef struct FcLayers {
	vector<FcParams> layerParams;
	int noLayers;
} FcLayers;
//--------------------------------------------------
void printFilter(Mat f) {
	for(int r=0; r < f.rows; r++) {
		for(int c=0; c<f.cols; c++) {
			cout << f.at<float>(r,c) << ",";
		}
		cout << endl;
	}
	cout << "-------------" << endl;
}
//--------------------------------------------------

void populateConvParams(ConvParams &cp, const float *w, const float *b) {

	float k;
	for(int n = 0; n < cp.noOutputs; n++) {
		vector<Mat> filt3D;
		for(int m = 0; m < cp.noInputs; m++) {
	        Mat filt2D = Mat(cp.filtH,cp.filtW, CV_32F);
			for(int i = 0; i < cp.filtH; i++) {
				for(int j = 0; j < cp.filtW; j++) {
					k = w[n*cp.noInputs*cp.filtH*cp.filtW+ m*cp.filtH*cp.filtW + i*cp.filtW + j];
					filt2D.at<float>(i, j) = k;
				}
			}
			filt3D.push_back(filt2D.clone());
		}
		cp.W.push_back(filt3D);
	}
	cp.bias = vector<float>(cp.noOutputs, 0);
	for(int n = 0; n < cp.noOutputs; n++) {
		cp.bias[n] = b[n];
	}
}
//--------------------------------------------------

void populateFcParams( FcParams &fp,  const float *w, const float *b) {
	fp.W = Mat(fp.noOutputs, fp.noInputs, CV_32F);
	fp.bias = vector<float>(fp.noOutputs, 0);

	for(int i = 0; i < fp.noOutputs; i++) {
		for(int j = 0; j < fp.noInputs; j++) {
			fp.W.at<float>(i, j) = w[i*fp.noInputs+j];
		}
		fp.bias[i] = b[i];
	}
}
//--------------------------------------------------

void InitLenet5Model(ConvLayers &conv, FcLayers &fc) {

	// Initialize conv layer parameters
	conv.noLayers = 2;
	ConvParams convParam;
	convParam.filtH = CONV1_FILTER_HEIGHT;
	convParam.filtW = CONV1_FILTER_WIDTH;
	convParam.noInputs = CONV1_NO_INPUTS;
	convParam.noOutputs = CONV1_NO_OUTPUTS;
	populateConvParams(convParam, (const float *)conv1_weights, (const float *)conv1_bias);
	conv.layerParams.push_back(convParam);

	convParam.W.clear();
	convParam.bias.clear();

	convParam.filtH = CONV2_FILTER_HEIGHT;
	convParam.filtW = CONV2_FILTER_WIDTH;
	convParam.noInputs = CONV2_NO_INPUTS;
	convParam.noOutputs = CONV2_NO_OUTPUTS;
	populateConvParams(convParam, (const float *)conv2_weights, (const float *)conv2_bias);
	conv.layerParams.push_back(convParam);

	// Initialize fully connected layer paramters
	fc.noLayers = 2;
	FcParams fcParam1, fcParam2;
	fcParam1.noInputs = IP1_NO_INPUTS;
	fcParam1.noOutputs = IP1_NO_OUTPUTS;
	populateFcParams(fcParam1, (const float *)ip1_weights, (const float *)ip1_bias);
	fc.layerParams.push_back(fcParam1);

	fcParam2.noInputs = IP2_NO_INPUTS;
	fcParam2.noOutputs = IP2_NO_OUTPUTS;
	populateFcParams(fcParam2, (const float *)ip2_weights, (const float *)ip2_bias);
	fc.layerParams.push_back(fcParam2);
	cout << fc.layerParams[0].bias[0] << endl;
	cout << fc.layerParams[1].bias[0] << endl;
	
}



int main(int argc, char **argv) {

	if(argc < 2) {
		cout<<"Please specify the image path"<<endl;
		exit(0);
	}

	Mat input = imread(argv[1], IMREAD_GRAYSCALE);

	//namedWindow("Input image", CV_WINDOW_AUTOSIZE);
	//imshow("Input image", input);

	// Model
	ConvLayers convModel;
	FcLayers fcModel;
	// Model initialization
	InitLenet5Model(convModel, fcModel);

	cout<<"Application complete"<<endl;
	//waitKey(0);
	return 0;
}
