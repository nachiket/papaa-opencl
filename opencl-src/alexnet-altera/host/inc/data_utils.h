#ifndef _DATA_UTILS_H_
#define _DATA_UTILS_H_

#include "cnn_structs.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef enum {
	CENTER,
	RAND
} CROP_TYPE_E;
template<typename T>
void showMat(T buff, int n_ch, int h, int w, int to_show=3);

Mat & cropImage(const Mat &img, unsigned int H, unsigned int W, CROP_TYPE_E type);

void initInputImage(const Mat &img, const Mat &mean, scoped_aligned_ptr<DTYPE> &h_input_img);

void zeropadAndTx(const scoped_aligned_ptr<DTYPE> &src, scoped_aligned_ptr<DTYPE> &dst,
	int n_ch, int src_h, int src_w, int pad_h, int pad_w, cl_mem &device_buff, bool h2d_tx);
#endif // _DATA_UTILS_H_
