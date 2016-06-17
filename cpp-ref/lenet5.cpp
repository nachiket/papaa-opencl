/*
Author: Gopalakrishna Hegde, NTU Singapore
Date: 17 June 2016


Handwritten digit recognition using Lenet5 CNN.
The network is trained using Caffe and the model is stored as C file
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
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
	
}

// Simple convolution(mathematically it is correlation). Assume no padding = 0
void convLayer(const vector<Mat> & input, vector<Mat> &output, const ConvParams &cp) {

	for(int n = 0 ;n < cp.noOutputs; n++) {
		Mat filt;
		Mat out = Mat::zeros(input[0].rows-cp.filtH+1, input[0].cols-cp.filtW+1, CV_32F);
		for(int m = 0; m < cp.noInputs; m++) {
			// 2D correlation
			filter2D(input[m], filt, -1, cp.W[n][m], Point(-1, -1), 0, BORDER_CONSTANT);
			// neglect the output corresponding to padded border
			filt = filt.colRange((cp.W[n][m].cols-1)/2, filt.cols - cp.W[n][m].cols/2).rowRange((cp.W[n][m].rows-1)/2, filt.rows -
				cp.W[n][m].rows/2);
			// add output from all input channels
			add(out, filt, out);
		}
		// add bias
		add(out, cp.bias[n], out);
		// store a deep copy
		output.push_back(out.clone());
	}
}

void maxPoolLayer(const vector<Mat> &input, vector<Mat> &output, int winSize, int stride) {
	int outH, outW;
	// this is going to be output map size
	outH = (input[0].rows - winSize ) / stride + 1;
	outW = (input[0].cols - winSize ) / stride + 1;

	Mat out = Mat::zeros(outH, outW, CV_32F);

	for(int n = 0; n < input.size(); n++) {
		for(int row = 0; row < input[n].rows; row += stride) {
			for(int col = 0; col < input[n].cols; col += stride) {
				Mat temp;
				Point minLoc, maxLoc;
				double minVal, maxVal;
				// this is the pooling window region
				Rect roi = Rect(col, row, winSize, winSize);
				input[n](roi).copyTo(temp);
				// find the max value in the pooling region
				minMaxLoc(temp, &minVal, &maxVal);
				// store the max value
				out.at<float>(row/stride, col/stride) = (float)maxVal;
			}
		}
		// store a deep copy
		output.push_back(out.clone());
	}
}


void reluLayer(vector<float> &input) {
	for(int e = 0; e < input.size(); e++){
		if(input[e] < 0.0) {
			input[e] = 0.0;
		}
	}
}

void innerProductLayer(const vector<float> &input, vector<float> &output, const FcParams &fp) {
	float acc;
	for(int n = 0; n < fp.noOutputs; n++) {
		acc = 0;
		// dot product
		for(int m = 0; m < fp.noInputs; m++) {
			acc += input[m] * fp.W.at<float>(n, m);
		}
		// add bias
		acc += fp.bias[n];
		// store the deep copy
		output.push_back(acc);
	}
}

void innerProductLayer(const vector<Mat> &input, vector<float> &output, const FcParams &fp) {

	// unroll all the feature maps
	// There should be a compact way to do this. Lets keep it open
	assert(fp.noInputs == input.size()*input[0].rows*input[0].cols);
	vector<float> inArray(fp.noInputs);

	for(int map = 0; map < input.size(); map++) {
		for(int row = 0; row < input[map].rows; row++) {
			for(int col = 0; col < input[map].cols; col++) {
				inArray[map*input[map].rows*input[map].cols + row*input[map].cols + col] = input[map].at<float>(row, col);
			}
		}
	}
	innerProductLayer(inArray, output, fp);
}

void softmaxLayer(vector<float> &input, vector<float> &output) {
	float sum = 0;
	for(int n = 0; n < input.size(); n++) {
		output.push_back(exp(input[n]));
		sum += output[n];
	}
	for(int n = 0; n < input.size(); n++) {
		output[n] /= sum;
	}
}

int lenet5App(Mat &input, const ConvLayers &convModel, const FcLayers &fcModel) {
	vector<Mat> inVec;
	vector<Mat> conv1Out, conv2Out, pool1Out, pool2Out;
	vector<float> ip1Out, ip2Out, prob;
	// conv layer requires vector of mat.
	inVec.push_back(input);
	// conv layer 1
	convLayer(inVec, conv1Out, convModel.layerParams[0]);
	// pool layer 1
	maxPoolLayer(conv1Out, pool1Out, 2, 2);
	// conv layer 2
	convLayer(pool1Out, conv2Out, convModel.layerParams[1]);
	// pool layer 2
	maxPoolLayer(conv2Out, pool2Out, 2, 2);

	// inner product 1
	innerProductLayer(pool2Out, ip1Out, fcModel.layerParams[0]);
	// ReLU layer
	reluLayer(ip1Out);
	// inner product 2
	innerProductLayer(ip1Out, ip2Out, fcModel.layerParams[1]);

	softmaxLayer(ip2Out, prob);

	cout << "-----------Output probabilities--------" << endl;
	for(int p = 0; p < prob.size(); p++) {
		cout << prob[p] << ",  ";
	}
	cout << endl << endl;
	
	return distance(prob.begin(), max_element(prob.begin(), prob.end()));

	
}
int main(int argc, char **argv) {

	if(argc < 2) {
		cout<<"Please specify the image path"<<endl;
		exit(0);
	}

	Mat input = imread(argv[1], IMREAD_GRAYSCALE);

	// Input normalization to make it in the range [0, 1], assuming the input image to be in the range [0, 255]
	Mat normInput;
	input.convertTo(normInput, CV_32F);
	normInput = normInput/255;

	// Model storage 
	ConvLayers convModel;
	FcLayers fcModel;

	// Model initialization
	InitLenet5Model(convModel, fcModel);

	// Forward pass of the network
	cout << "Starting the forward pass...." << endl;
	int predNo = lenet5App(normInput, convModel, fcModel);

	cout << "The digit in the image is = " << predNo << endl;
	cout<<"Application complete"<<endl;

	return 0;
}
