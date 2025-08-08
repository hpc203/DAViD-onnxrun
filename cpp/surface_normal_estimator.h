# ifndef Surface_NormalEstimator_H
# define Surface_NormalEstimator_H
#include "pixelwise_estimator.h"


class SurfaceNormalEstimator : PixelwiseEstimator
{
public:
    SurfaceNormalEstimator(std::string onnxpath):PixelwiseEstimator(onnxpath)
    {};

    cv::Mat estimate_normal(const cv::Mat& image);

};


#endif