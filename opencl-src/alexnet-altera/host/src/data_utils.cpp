#include "data_utils.h"
#include <iostream>


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

void cropImage(const cv::Mat &img, cv::Mat &crop_img, unsigned int H, unsigned int W, CROP_TYPE_E type) {
		switch(type) {
			case CENTER:
			{
				int top_x = (img.cols - W)/2;
				int top_y = (img.rows - W)/2;
				cv::Rect window(top_x, top_y, W, H);
				crop_img = img(window);
				break;
			}
			case RAND:
				std::cout << "Not implemented" << std::endl;
				exit(1);
				break;
			default:
				std::cout << "Invalid crop type" << std::endl;
				exit(1);
		}
}

void initInputImage(const cv::Mat &img, const cv::Mat &mean, aocl_utils::scoped_aligned_ptr<DTYPE> &h_input_img) {
    uint8_t *p_img = (uint8_t *)img.data;
	uint8_t *p_mean = (uint8_t *)mean.data;
    uint8_t r,g,b, rm, gm, bm;
	assert(img.channels() == mean.channels() && img.rows == mean.rows && img.cols == mean.cols);
    unsigned int C = img.channels();
	unsigned int W = img.cols;
	unsigned int H = img.rows;
    for(int row = 0; row < H; row++) {
        for(int col = 0; col < W; col++) {
            b = p_img[row*W*C + col*C + 0];
            g = p_img[row*W*C + col*C + 1];
            r = p_img[row*W*C + col*C + 2];
            bm = p_mean[row*W*C + col*C + 0];
            gm = p_mean[row*W*C + col*C + 1];
            rm = p_mean[row*W*C + col*C + 2];
			h_input_img[0*H*W + row*W + col] = (DTYPE)(r - rm);
			h_input_img[1*H*W + row*W + col] = (DTYPE)(g - rm);
			h_input_img[2*H*W + row*W + col] = (DTYPE)(b - rm);
        }
    }
}

void zeropadAndTx(const aocl_utils::scoped_aligned_ptr<DTYPE> &src, aocl_utils::scoped_aligned_ptr<DTYPE> &dst,
	int n_ch, int src_h, int src_w, int pad_h, int pad_w, cl_mem &device_buff, cl_command_queue &queue, bool h2d_tx) {

	unsigned dst_h = src_h + 2*pad_h;
	unsigned dst_w = src_w + 2*pad_w;
	cl_int status;
	for(int ch = 0; ch < n_ch; ch++) {
		for(int r = 0; r < src_h + 2*pad_h; r++) {
			for(int c = 0; c < src_w + 2*pad_w; c++) {
				if(r < pad_h || (r > src_h+pad_h-1) || c < pad_w || (c > src_w+pad_w-1)) {
					dst[ch*dst_h*dst_w + r*dst_w + c] = (DTYPE)0;
				}else {
					// TODO: use memcpy instead to do bulk copy of one full row
					dst[ch*dst_h*dst_w + r*dst_w + c] = src[ch*src_h*src_w + (r-pad_h)*src_w + c-pad_w];
				}
			}	
		}
	}
	if(h2d_tx) {
		//std::cout << "Sending data to device buffer" << std::endl;
		status = clEnqueueWriteBuffer(queue, device_buff, CL_FALSE, 0,
			n_ch * dst_h * dst_w * sizeof(DTYPE), dst, 0, NULL, NULL);
		checkError(status, "Failed to transfer data to the device\n");
		clFinish(queue);
	}
}
