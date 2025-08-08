# ifndef MultiTask_Estimator_H
# define MultiTask_Estimator_H
#include "pixelwise_estimator.h"
#include <map>


class MultiTaskEstimator : PixelwiseEstimator
{
public:
    MultiTaskEstimator(std::string onnxpath, const bool is_inverse_depth=true, const float binarization_threshold=0):PixelwiseEstimator(onnxpath)
    {
        this->is_inverse_depth = is_inverse_depth; 
        this->binarization_threshold = binarization_threshold;
    };

    std::map<std::string, cv::Mat> estimate_all_tasks(const cv::Mat& image);
private:
    bool is_inverse_depth;
    float binarization_threshold;
};


#endif