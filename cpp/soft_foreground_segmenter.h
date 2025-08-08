# ifndef Soft_ForegroundSegmenter_H
# define Soft_ForegroundSegmenter_H
#include "pixelwise_estimator.h"


class SoftForegroundSegmenter : PixelwiseEstimator
{
public:
    SoftForegroundSegmenter(std::string onnxpath, const float binarization_threshold=0):PixelwiseEstimator(onnxpath)
    {this->binarization_threshold = binarization_threshold; };

    cv::Mat estimate_foreground_segmentation(const cv::Mat& image);
private:
    float binarization_threshold;
};


#endif