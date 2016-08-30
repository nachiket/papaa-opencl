#ifndef _DATA_UTILS_H_
#define _DATA_UTILS_H_
#include <iostream>
#include "cnn_structs.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CHECK_NEAR(val1, val2) \
		if(abs(val1 - val2) > 1e-6){ \
			std::cout << "Mismatch" << std::endl;\
		} \

typedef enum {
	CENTER,
	RAND
} CROP_TYPE_E;

cv::Mat & cropImage(const cv::Mat &img, unsigned int H, unsigned int W, CROP_TYPE_E type);

void initInputImage(const cv::Mat &img, DTYPE *mean, aocl_utils::scoped_aligned_ptr<DTYPE> &h_input_img);

void zeropadAndTx(const aocl_utils::scoped_aligned_ptr<DTYPE> &src, aocl_utils::scoped_aligned_ptr<DTYPE> &dst,
	int n_ch, int src_h, int src_w, int pad_h, int pad_w, cl_mem &device_buff, cl_command_queue &queue, bool h2d_tx);

void rand_init(aocl_utils::scoped_aligned_ptr<DTYPE> &buff, int len, int seed = 1234);

void rand_init(WTYPE *buff, int len, int seed = 1234);

template<typename T>
void img2col(const T &img, T &mat,
	const int n_ch, const int height, const int width,
	const int K, const int stride, const int pad, const int zero_rows, const int zero_cols);

template<typename T>
void col2img(const T & mat, T &img, int no_out, int out_h, int out_w, int no_pad_col);

void weight2col(const WTYPE *W, WTYPE *wmat, const int n_ch, const int K,
    const int no_filters, const int zero_rows, const int zero_cols);

template<typename T>
void showMat(T buff, int n_ch, int h, int w, int to_show=3) {
	for(unsigned int ch = 0; ch < to_show; ch++) {
		std::cout << "Channel: " << ch << std::endl;
		for(unsigned int r = 0; r < h; r++) {
			for(unsigned int c = 0; c < w; c++) {
				std::cout << buff[ch*h*w+r*w+c] << ",";
			}
			std::cout << std::endl;
		}
	} 
}
#endif // _DATA_UTILS_H_
