# ifndef Relative_DepthEstimator_H
# define Relative_DepthEstimator_H
#include "pixelwise_estimator.h"


class RelativeDepthEstimator : PixelwiseEstimator
{
public:
    RelativeDepthEstimator(std::string onnxpath, const bool is_inverse=true):PixelwiseEstimator(onnxpath)
    {this->is_inverse = is_inverse; };

    cv::Mat estimate_relative_depth(const cv::Mat& image);
private:
    bool is_inverse;
};


#endif