#pragma once
#include<opencv\cv.h>
using namespace std;

class SkinDetector
{
    public:
     SkinDetector(void);
    ~SkinDetector(void);

cv::Mat getSkin(cv::Mat input);

private:
int Y_MIN;
int Y_MAX;
int Cr_MIN;
int Cr_MAX;
int Cb_MIN;
int Cb_MAX;
};

//SkinDetector.cpp

#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

SkinDetector::SkinDetector(void)
{
    Y_MIN  = 0;
    Y_MAX  = 255;
    Cr_MIN = 133;
    Cr_MAX = 173;
    Cb_MIN = 77;
    Cb_MAX = 127;
}

SkinDetector::~SkinDetector(void)
{
}

cv::Mat SkinDetector::getSkin(cv::Mat input)
{
cv::Mat skin;
cv::cvtColor(input,skin,cv::COLOR_BGR2YCrCb);

cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);

return skin;
}
