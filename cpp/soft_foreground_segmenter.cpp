#include "soft_foreground_segmenter.h"


using namespace cv;
using namespace std;


Mat SoftForegroundSegmenter::estimate_foreground_segmentation(const Mat& image)
{
    std::tuple<std::vector<cv::Mat>, MetaData_> result = this->_estimate_dense_map(image);
    Mat mask = std::get<0>(result)[0];

    Mat segmented_image = composite_model_output_to_image(mask, std::get<1>(result), cv::INTER_CUBIC);
    segmented_image.setTo(0, segmented_image<0);
    segmented_image.setTo(1, segmented_image>1);
    if(this->binarization_threshold>0)
    {
        Mat result = cv::Mat::zeros(segmented_image.rows, segmented_image.cols, CV_32FC1);
        result.setTo(1, segmented_image>this->binarization_threshold);
        result.convertTo(result, CV_8UC1);
        return result;
    }
    return segmented_image;
}