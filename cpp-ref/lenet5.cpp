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
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include "lenet5_model.h"
#include "papi.h"

using namespace cv;
using namespace std;

unsigned long long t1, t2;

// Structure defs to store the layer weights and bias
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

// Just for debugging purpose
//--------------------------------------------------
void printFilter(Mat f) {
	for(int r=0; r < f.rows; r++) {
		for(int c=0; c<f.cols; c++) {
			//cout << f.at<float>(r,c) << ",";
			printf("%1.2f,",f.at<float>(r,c));
		}
		cout << endl;
	}
	cout << "-------------" << endl;
}

//--------------------------------------------------

// Initialize the convolution layer parameters from the model file.
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

// Initialize Lenet5 model
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

//--------------------------------------------------
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


//--------------------------------------------------
// Feature map accumulator. basically to perform accumulation in the 3rd dimension
void mapAdd(const Mat &map, Mat &acc) {
	assert(map.size() == acc.size());
	for(int r = 0; r < map.rows; r++) {
		for(int c = 0; c < map.cols; c++) {
			acc.at<float>(r,c) += map.at<float>(r, c);
		}
	}
}


// Enable this to use cv::filter2D
//#define USE_OPENCV_FILTER2D
// Simple convolution(mathematically it is correlation). Assume no padding = 0
void convLayer(const vector<Mat> & input, vector<Mat> &output, const ConvParams &cp) {

	Mat filt = Mat(input[0].rows-cp.filtH+1, input[0].cols-cp.filtW+1, CV_32F);
	for(int n = 0 ;n < cp.noOutputs; n++) {
		Mat out = Mat::zeros(input[0].rows-cp.filtH+1, input[0].cols-cp.filtW+1, CV_32F);
		for(int m = 0; m < cp.noInputs; m++) {
#ifdef USE_OPENCV_FILTER2D
			// 2D correlation
			filter2D(input[m], filt, -1, cp.W[n][m], Point(-1, -1), 0, BORDER_CONSTANT);
			// neglect the output corresponding to padded border
			filt = filt.colRange((cp.W[n][m].cols-1)/2, filt.cols - cp.W[n][m].cols/2).rowRange((cp.W[n][m].rows-1)/2, filt.rows -
				cp.W[n][m].rows/2);
			// add output from all input channels
			add(out, filt, out);
#else
			customFilter2D(input[m], filt, cp.W[n][m]);
			mapAdd(filt, out);
#endif // USE_OPENCV_FILTER2D
		}
		// add bias
#ifdef USE_OPENCV_FILTER2D
		add(out, cp.bias[n], out);
#else
		for(int r = 0; r < out.rows; r++) {
			for(int c = 0; c < out.cols; c++) {
				out.at<float>(r,c) += cp.bias[n];
			}
		}
#endif
		// store a deep copy
		output.push_back(out.clone());
	}
}


//--------------------------------------------------
// max pooling.
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


//--------------------------------------------------
// Rectification layer
void reluLayer(vector<float> &input) {
	for(int e = 0; e < input.size(); e++){
		if(input[e] < 0.0) {
			input[e] = 0.0;
		}
	}
}


//--------------------------------------------------
// Fully connected layer of a neural net. 
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


//--------------------------------------------------
// Variant of inner product layer whose input is set of feature maps instead of a single feature vector.
// ie. Input is pool or conv layer
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
	// It is the normal inner product once we unroll the feature maps into a feature vector.
	innerProductLayer(inArray, output, fp);
}


//--------------------------------------------------
// Softmax probability layer.
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


//--------------------------------------------------
// Here we call all layer APIs in sequence to get the final prediction
int lenet5App(Mat &input, const ConvLayers &convModel, const FcLayers &fcModel) {
	vector<Mat> inVec;
	vector<Mat> conv1Out, conv2Out, pool1Out, pool2Out;
	vector<float> ip1Out, ip2Out, prob;
	// conv layer requires vector of mat.
	inVec.push_back(input);
	
	t1 = PAPI_get_virt_usec();
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

	t2 = PAPI_get_virt_usec();

	cout << "-----------Output probabilities--------" << endl;
	for(int p = 0; p < prob.size(); p++) {
		cout << prob[p] << ",  ";
	}
	cout << endl << endl;
	
	return distance(prob.begin(), max_element(prob.begin(), prob.end()));

	
}

void print_help(char **argv) {
	printf("Usage : %s\n"
		"-m sample -i <image path>\n"
		"\tOR\t\n"
		"-m test -f <image list file> -d <image dir> [-n <no images to test>]\n",argv[0]);
}

int main(int argc, char **argv) {

	char * mode = NULL;
	char * imgName = NULL;
	char * imgListFile = NULL;
	char * imgDir = NULL;
	int noTestImgs = -1;
	if(argc == 1) {
		print_help(argv);
		return -1;
	}

	// parse arguments and decide the application mode.
	for(int i = 1; i < argc; i++) {
		if(!strcmp(argv[i], "-m")) {
			mode = argv[++i];
		} else if (!strcmp(argv[i], "-i")){
			imgName = argv[++i];
		} else if(!strcmp(argv[i], "-f")) {
			imgListFile = argv[++i];
		} else if(!strcmp(argv[i], "-d")) {
			imgDir = argv[++i];
		} else if(!strcmp(argv[i], "-n")) {
			noTestImgs = atoi(argv[++i]);
		}
	}

	// Model storage 
	ConvLayers convModel;
	FcLayers fcModel;

	// Model initialization
	InitLenet5Model(convModel, fcModel);

	if(!strcmp(mode, "sample")) {

		Mat input = imread(imgName, IMREAD_GRAYSCALE);

		// Input normalization to make it in the range [0, 1], assuming the input image to be in the range [0, 255]
		Mat normInput;
		input.convertTo(normInput, CV_32F);
		normInput = normInput/255;


		// Forward pass of the network
		cout << "Starting the forward pass...." << endl;
		int predNo = lenet5App(normInput, convModel, fcModel);
		cout << "The digit in the image is = " << predNo << endl;
		cout << "Application runtime(usec) = " << (t2-t1) <<endl;

	} else if(!strcmp(mode, "test")) {
		cout << "********MNIST Test Mode*********" << endl;

		std::ifstream listFile;
		std::vector<std::string> testImageList;
		std::vector<int> targetLabels, predLabels;
		std::string csvLine, imgFile, label;

		// read image list file and target labels and store in a vector
		listFile.open(imgListFile);
		while(std::getline(listFile, csvLine)) {
			std::istringstream ss(csvLine);
			std::getline(ss, imgFile, ',');
			std::getline(ss, label, ',');
			testImageList.push_back(imgFile);
			targetLabels.push_back(atoi(label.c_str()));
		}
		cout << "No of test images = " << targetLabels.size() << endl;
		int pred;
		int misCount = 0;
		Mat input, normInput;

		// This is the directory containing all MNIST test images.
		std::string testImgDir(imgDir);
		if(noTestImgs < 0) {
			noTestImgs = targetLabels.size();
		}

		for (int im = 0; im < noTestImgs; im++) {
			imgFile = testImgDir + "/" + testImageList[im];
			cout << imgFile << endl;
			input = imread(imgFile, IMREAD_GRAYSCALE);

			// Input normalization to make it in the range [0, 1], assuming the input image to be in the range [0, 255]
			//Mat normInput;
			input.convertTo(normInput, CV_32F);
			normInput = normInput/255;

			// Prediction of digit
			pred = lenet5App(normInput, convModel, fcModel);

			// check if the computer got it right...
			if(pred != targetLabels[im]) {
				misCount++;
			}
		}
		cout << "No images misclassified = " << misCount << endl;
		cout << "Classification Error = " << float(misCount)/noTestImgs << endl;
		
		
	} else {
		cout << "Invalid application mode" << endl;
		return -1;
	}


	return 0;
}
