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
#include<time.h>
#include <sys/time.h>
#include <unistd.h>


#define SAMPLE_SIZE 20
#define OVERLAP_SIZE 5
#define OUTPUTX_SIZE 250
#define OUTPUTY_SIZE 250

using std::cout;
using std::endl;

std::vector<cv::Mat> createImageList(cv::Mat& hSrc) {
	int x_size = hSrc.rows;
	int y_size = hSrc.cols;
	std::vector<cv::Mat> imglist((x_size - SAMPLE_SIZE) * (y_size - SAMPLE_SIZE));
	for(int i = 0; i < x_size - SAMPLE_SIZE; i++) {
		for(int j = 0; j < y_size - SAMPLE_SIZE; j++) {
			imglist[(i * (y_size - SAMPLE_SIZE)) + j] = hSrc(cv::Range(i, i + SAMPLE_SIZE), cv::Range(j, j + SAMPLE_SIZE));
		}
	}
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

int computeHorizontalSSD(cv::Mat& topImg, cv::Mat& randImg) {
	if(topImg.dims == 0) {
		return 0;
	}
	cv::Mat overlap1 = topImg(cv::Range(topImg.rows-OVERLAP_SIZE, topImg.rows), cv::Range(0,topImg.cols));
	cv::Mat overlap2 = randImg(cv::Range(0, OVERLAP_SIZE), cv::Range(0,topImg.cols));
	return computeSSD(overlap1, overlap2);
}

int computeVerticalSSD(cv::Mat& prevImg, cv::Mat& randImg) {
	if(prevImg.dims == 0) {
		return 0;
	}
	cv::Mat overlap1 = prevImg(cv::Range(0,prevImg.rows),cv::Range(prevImg.cols-OVERLAP_SIZE,prevImg.cols));
	cv::Mat overlap2 = randImg(cv::Range(0,randImg.rows),cv::Range(0,OVERLAP_SIZE));
	return computeSSD(overlap1, overlap2);
}

int computeCombinedSSD(cv::Mat& prevImg, cv::Mat& topImg, cv::Mat& randImg) {
	double verticalSSD = computeVerticalSSD(prevImg, randImg);
	double horizontalSSD = computeHorizontalSSD(topImg, randImg);
	return verticalSSD + horizontalSSD;
}

cv::Mat getMinSSDImg(cv::Mat& prevImg, cv::Mat& topImg, std::vector<cv::Mat>& imglist) {
	int minSSD = 0;
	int minIdx = 0;
	for(int i = 0; i < imglist.size(); i++) {
		if(i == 0) {
			minSSD = computeCombinedSSD(prevImg, topImg, imglist[i]);
			minIdx = i;
		} else {
			int ssd = computeCombinedSSD(prevImg, topImg, imglist[i]);
			if(ssd < minSSD) {
				minSSD = ssd;
				minIdx = i;
			}
		}
	}
	//cout << minSSD << "\n" << endl;
	//cout << minIdx << endl;
	return imglist[minIdx];
}

cv::Mat getPreviousImg(int i, int j, cv::Mat& hDst) {
	cv::Mat subImg;
	if(j == 0) {
		return subImg;
	} else {
		subImg = hDst(cv::Range((i*SAMPLE_SIZE)-(OVERLAP_SIZE*i),((i+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*i)),cv::Range(((j-1)*SAMPLE_SIZE)-(j-1)*OVERLAP_SIZE,(j*SAMPLE_SIZE)-(j-1)*OVERLAP_SIZE));
		return subImg;
	}
}

cv::Mat getTopImg(int i, int j, cv::Mat& hDst) {
	cv::Mat subImg;
	if(i == 0) {
		return subImg;
	} else {
		subImg = hDst(cv::Range(((i-1)*SAMPLE_SIZE)-(i-1)*OVERLAP_SIZE,(i*SAMPLE_SIZE)-(i-1)*OVERLAP_SIZE),cv::Range((j*SAMPLE_SIZE)-(OVERLAP_SIZE*j),((j+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*j)));
		return subImg;
	}
}

void placeImg(int row, int col, cv::Mat& tile, cv::Mat& lImg) {
	int x1 = (row*SAMPLE_SIZE)-(OVERLAP_SIZE*row);
	int x2 = ((row+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*row);
	int y1 = (col*SAMPLE_SIZE)-(OVERLAP_SIZE*col);
	int y2 = ((col+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*col);
	if(row == 0) {
		x1 = (row*SAMPLE_SIZE);
		x2 = ((row+1)*SAMPLE_SIZE);
	}
	if(col == 0){
		y1 = (col*SAMPLE_SIZE);
		y2 = ((col+1)*SAMPLE_SIZE);
	}
	//cout << "inside placeImage" << x1 << ":" << x2 << ":" << y1 << ":" << y2 << ":" << endl;
	tile.copyTo(lImg(cv::Range(x1, x2), cv::Range(y1, y2)));
}

void imageQuilting(cv::Mat& hSrc, cv::Mat& hDst) {


	std::vector<cv::Mat> imglist = createImageList(hSrc);

	int nx = OUTPUTX_SIZE/(SAMPLE_SIZE - OVERLAP_SIZE);
	int ny = OUTPUTY_SIZE/(SAMPLE_SIZE - OVERLAP_SIZE);

	for(int i = 0; i < nx; i++ ) {
		for(int j = 0; j < ny; j++) {

			//cout << "i , j : " << i << " : " << j << endl;
			cv::Mat prevImg = getPreviousImg(i, j, hDst);

			cv::Mat topImg = getTopImg(i, j, hDst);

			clock_t start = clock();
			cv::Mat currImg = getMinSSDImg(prevImg, topImg, imglist);
			clock_t end = clock();
			double elapsed_secs = double(end - start) * 1000 / CLOCKS_PER_SEC;

			cout << "Execution time:" << elapsed_secs <<"  ms"<< endl;

			//cout << currImg << endl;
			placeImg(i, j, currImg, hDst);

		}
	}

	//dSrc.copyTo(dDst);
	//dDst.download(hDst);
}

int main() {

	std::cout << "Hello World" << std::endl;
	std::string imageName = "image1.png";
	cv::Mat input = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);

	//cv::Mat inputGray = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);


	//std::cout << "gray :\n" << inputGray << endl;

	if (input.empty()) {
		cout << "Cannot read " + imageName << endl;
	} else {
		cout << imageName + " loaded" << endl;
	}
	cv::Mat output(OUTPUTY_SIZE, OUTPUTX_SIZE, CV_8UC3);

	clock_t start = clock();
	imageQuilting(input, output);
	clock_t end = clock();
	double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

	cout << "time elapsed :" << elapsed_secs << endl;

	cv::imshow("Output", output);

	cv::waitKey();

	return 0;
}
