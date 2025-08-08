#include "depth_estimator.h"


using namespace cv;
using namespace std;


Mat RelativeDepthEstimator::estimate_relative_depth(const Mat& image)
{
    std::tuple<std::vector<cv::Mat>, MetaData_> result = this->_estimate_dense_map(image);
    Mat depth = std::get<0>(result)[0];
    Mat depth_map = composite_model_output_to_image(depth, std::get<1>(result), cv::INTER_CUBIC);
    if(this->is_inverse)
    {
        depth_map *= -1;
    }
    return depth_map;
}