/*
 * ImgQltService.cu
 *
 *  Created on: 07-Oct-2015
 *      Author: mobin
 */

#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include"opencv2/highgui/highgui.hpp"
#include<cuda_runtime.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include"opencv2/cudaarithm.hpp"
#include "tbb/tbb_stddef.h"

#define BLOCK_SIZE 16

using std::cout;
using std::endl;
using namespace cv::cuda;

int outputX_size = 250;
int outputY_size = 250;
int sample_size = 20;
int overlap_size = 5;

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int sample_size, size_t step)
{
   Matrix Asub;
   Asub.width    = BLOCK_SIZE;
   Asub.height   = BLOCK_SIZE;
   Asub.stride   = A.stride;
   Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                        + BLOCK_SIZE * col];
   return Asub;
}

__global__ void cudaCreateImageList(uchar* dSrc, std::vector<cv::cuda::GpuMat>& dDst, int height, int width, int sample_size, int step) {

	int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;

	printf("\n indices : %d %d",rowIdx, colIdx);
	printf("\nb : %u g : %u r : %u",dSrc[(step*rowIdx+3*colIdx)],dSrc[(step*rowIdx+3*colIdx)+1],dSrc[(step*rowIdx+3*colIdx)+2]);
	//dDst[(xIndex * height) + yIndex] = dSrc;//dSrc(cv::Range(i, i + sample_size), cv::Range(j, j + sample_size));
}

__global__ void copyImg(uchar* dSrc, uchar* dDst, int height, int width, int sample_size, int step) {

	int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;

	dDst[(step*rowIdx+3*colIdx)]=dSrc[(step*rowIdx+3*colIdx)];
	dDst[(step*rowIdx+3*colIdx)+1]=dSrc[(step*rowIdx+3*colIdx)+1];
	dDst[(step*rowIdx+3*colIdx)+2]=dSrc[(step*rowIdx+3*colIdx)+2];
	//dDst[(xIndex * height) + yIndex] = dSrc;//dSrc(cv::Range(i, i + sample_size), cv::Range(j, j + sample_size));
}

std::vector<cv::Mat> createImageList(cv::Mat& hSrc) {
	int height = hSrc.rows;
	int width = hSrc.cols;

	cv::cuda::GpuMat dSrc, dDst(height, width, CV_8UC3);
	dSrc.upload(hSrc);

	std::vector<cv::Mat> imglist((height - sample_size) * (width - sample_size));
	for(int i = 0; i < height - sample_size; i++) {
		for(int j = 0; j < width - sample_size; j++) {
			imglist[(i * (width - sample_size)) + j] = hSrc(cv::Range(i, i + sample_size), cv::Range(j, j + sample_size));
		}
	}


	std::vector<cv::cuda::GpuMat> dList((height - sample_size) * (width - sample_size));

	const dim3 grid(8,8);
	const dim3 block(((width - sample_size)/grid.x)+1,((height - sample_size)/grid.y)+1);

	copyImg<<<grid,block>>>(dSrc.ptr(),dDst.ptr(),height-sample_size,width-sample_size,sample_size,dSrc.step);

	cout << "\ninput" << hSrc << endl;
	cv::Mat output;
	dDst.download(output);
	cout << "\noutput" << output << endl;
	cv::imshow("Output", output);
	//cudaCreateImageList<<<grid,block>>>(dSrc.ptr(),dList,height-sample_size,width-sample_size,sample_size,dSrc.step);

	return imglist;
}

double getPixelValue(cv::Vec3b& pixel) {
	int b = pixel[0];
	int g = pixel[1];
	int r = pixel[2];

	//cout << "b " << b << "g " << g << "r "<< r <<  endl;
	//cout << "result:" << 0.2989 * r + 0.5870 * g + 0.1140 * b << endl;
	return 0.2989 * r + 0.5870 * g + 0.1140 * b;
}

int computeSSD(cv::Mat& overlap1, cv::Mat& overlap2) {
	double sum = 0;
	for (int i = 0; i < overlap1.rows; i++) {
		for (int j = 0; j < overlap1.cols; j++) {
			double val1 = getPixelValue(overlap1.at<cv::Vec3b>(i,j));
			//cout << "val1: " << val1 << endl;
			double val2 = getPixelValue(overlap2.at<cv::Vec3b>(i,j));
			//cout << "val2: " << val2 << endl;
			sum += std::sqrt(std::pow((val1 - val2), 2 ));
			//cout << "sum " << sum << endl;
		}
		//	sum += ((oi1[j] - oi2[j]) ^ 2) ^ 0.5;
	}

	//cout << "overlap2.rows " << overlap1.rows << endl;
	//cout << "overlap2.cols " << overlap1.cols << endl;

	return sum;
}

int computeHorizontalSSD(cv::Mat& topImg, cv::Mat& randImg, int overlap_size) {
	if(topImg.dims == 0) {
		return 0;
	}
	cv::Mat overlap1 = topImg(cv::Range(topImg.rows-overlap_size, topImg.rows), cv::Range(0,topImg.cols));
	cv::Mat overlap2 = randImg(cv::Range(0, overlap_size), cv::Range(0,topImg.cols));
	return computeSSD(overlap1, overlap2);
}

int computeVerticalSSD(cv::Mat& prevImg, cv::Mat& randImg, int overlap_size) {
	if(prevImg.dims == 0) {
		return 0;
	}
	cv::Mat overlap1 = prevImg(cv::Range(0,prevImg.rows),cv::Range(prevImg.cols-overlap_size,prevImg.cols));
	cv::Mat overlap2 = randImg(cv::Range(0,randImg.rows),cv::Range(0,overlap_size));
	return computeSSD(overlap1, overlap2);
}

int computeCombinedSSD(cv::Mat& prevImg, cv::Mat& topImg, cv::Mat& randImg, int overlap_size) {
	double verticalSSD = computeVerticalSSD(prevImg, randImg, overlap_size);
	double horizontalSSD = computeHorizontalSSD(topImg, randImg, overlap_size);
	return verticalSSD + horizontalSSD;
}

cv::Mat getMinSSDImg(cv::Mat& prevImg, cv::Mat& topImg, std::vector<cv::Mat>& imglist) {
	int minSSD = 0;
	int minIdx = 0;
	for(int i = 0; i < imglist.size(); i++) {
		if(i == 0) {
			minSSD = computeCombinedSSD(prevImg, topImg, imglist[i], overlap_size);
			minIdx = i;
		} else {
			int ssd = computeCombinedSSD(prevImg, topImg, imglist[i], overlap_size);
			if(ssd < minSSD) {
				minSSD = ssd;
				minIdx = i;
			}
		}
	}
	return imglist[minIdx];
}

cv::Mat getPreviousImg(int i, int j, cv::Mat& hDst) {
	cv::Mat subImg;
	if(j == 0) {
		return subImg;
	} else {
		subImg = hDst(cv::Range((i*sample_size)-(overlap_size*i),((i+1)*sample_size)-(overlap_size*i)),cv::Range(((j-1)*sample_size)-(j-1)*overlap_size,(j*sample_size)-(j-1)*overlap_size));
		return subImg;
	}
}

cv::Mat getTopImg(int i, int j, cv::Mat& hDst) {
	cv::Mat subImg;
	if(i == 0) {
		return subImg;
	} else {
		subImg = hDst(cv::Range(((i-1)*sample_size)-(i-1)*overlap_size,(i*sample_size)-(i-1)*overlap_size),cv::Range((j*sample_size)-(overlap_size*j),((j+1)*sample_size)-(overlap_size*j)));
		return subImg;
	}
}

void placeImg(int row, int col, cv::Mat& tile, cv::Mat& lImg) {
	int x1 = (row*sample_size)-(overlap_size*row);
	int x2 = ((row+1)*sample_size)-(overlap_size*row);
	int y1 = (col*sample_size)-(overlap_size*col);
	int y2 = ((col+1)*sample_size)-(overlap_size*col);
	if(row == 0) {
		x1 = (row*sample_size);
		x2 = ((row+1)*sample_size);
	}
	if(col == 0){
		y1 = (col*sample_size);
		y2 = ((col+1)*sample_size);
	}

	tile.copyTo(lImg(cv::Range(x1, x2), cv::Range(y1, y2)));
}

void imageQuilting(cv::Mat& hSrc, cv::Mat& hDst) {

	int height = hSrc.rows;
	int width = hSrc.cols;
	//std::cout << "inside image quilting" << endl;

	cv::cuda::GpuMat dDst;

	std::vector<cv::Mat> imglist = createImageList(hSrc);

	int nx = outputX_size/(sample_size - overlap_size);
	int ny = outputY_size/(sample_size - overlap_size);
	int newx = nx + (height - nx * overlap_size) / sample_size;
	int newy = ny + (width - ny * overlap_size) / sample_size;

	for(int i = 0; i < newx; i++ ) {
		for(int j = 0; j < newy; j++) {
			//cout << "i , j : " << i << " : " << j << endl;

			cv::Mat prevImg = getPreviousImg(i, j, hDst);

			cv::Mat topImg = getTopImg(i, j, hDst);

			cv::Mat currImg = getMinSSDImg(prevImg, topImg, imglist);
			placeImg(i, j, currImg, hDst);

		}
	}

	//dSrc.copyTo(dDst);
	//dDst.download(hDst);
}

int main() {


	int num_devices = getCudaEnabledDeviceCount();
	cout << "gpu count :" << num_devices << endl;

	std::cout << "Hello World" << std::endl;
	std::string imageName = "image1.png";
	cv::Mat input = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);

	if (input.empty()) {
		cout << "Cannot read " + imageName << endl;
	} else {
		cout << imageName + " loaded" << endl;
		/*cout << input << endl;
		int b = input.at<cv::Vec3b>(0,0)[0];
		int g = input.at<cv::Vec3b>(0,0)[1];
		int r = input.at<cv::Vec3b>(0,0)[2];

		int val = input.at<int>(0,0);
		cout << b << g << r <<  endl;*/
	}
	cv::Mat output(outputY_size, outputX_size, CV_8UC3);

	imageQuilting(input, output);

	//cv::imshow("Output", output);

	cv::waitKey();

	return 0;
}
