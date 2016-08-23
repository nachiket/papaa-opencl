#include "data_utils.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"

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

void initInputImage(const cv::Mat &img, DTYPE *mean, aocl_utils::scoped_aligned_ptr<DTYPE> &h_input_img) {
	// Resize image to be 256x256
	cv::Mat res_img;
	cv::Size size(256, 256);
	cv::resize(img, res_img, size);
	// mean normalization
	cv::Mat mean_norm = cv::Mat::zeros(size, CV_32FC3);
	DTYPE rm, gm, bm;
	cv::Vec3b pixel_u8;
	cv::Vec3f pixel_f;
	for(int r = 0; r < res_img.rows; r++) {
		for(int c = 0; c < res_img.cols; c++) {
			pixel_u8 = res_img.at<cv::Vec3b>(r,c);
			rm = mean[0*256*256+r*256+c];
			gm = mean[1*256*256+r*256+c];
			bm = mean[2*256*256+r*256+c];
			pixel_f.val[0] = pixel_u8.val[0] - bm;
			pixel_f.val[1] = pixel_u8.val[1] - gm;
			pixel_f.val[2] = pixel_u8.val[2] - rm;
			mean_norm.at<cv::Vec3f>(r,c) = pixel_f;
		}
	}	
	cv::Mat crop_img;

	cropImage(mean_norm, crop_img, 227, 227, CENTER);
    unsigned int C = crop_img.channels();
	unsigned int W = crop_img.cols;
	unsigned int H = crop_img.rows;
    for(int row = 0; row < H; row++) {
        for(int col = 0; col < W; col++) {
			pixel_f = crop_img.at<cv::Vec3f>(row,col);
			// BGR to RGB conversion will happen here by properly storing pixels in that manner.
			h_input_img[0*H*W + row*W + col] = pixel_f[2];
			h_input_img[1*H*W + row*W + col] = pixel_f[1];
			h_input_img[2*H*W + row*W + col] = pixel_f[0];
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
