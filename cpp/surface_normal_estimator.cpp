#include "surface_normal_estimator.h"


using namespace cv;
using namespace std;


Mat SurfaceNormalEstimator::estimate_normal(const Mat& image)
{
    std::tuple<std::vector<cv::Mat>, MetaData_> result = this->_estimate_dense_map(image);
    Mat normal = std::get<0>(result)[0];

    Mat normal_map = composite_model_output_to_image(normal, std::get<1>(result), cv::INTER_CUBIC);
    ////拆分通道
    std::vector<Mat> channels;
    cv::split(normal_map, channels);
    ////向量化计算范数平方
    Mat bSq, gSq, rSq;
    cv::pow(channels[0], 2, bSq);
    cv::pow(channels[1], 2, gSq);
    cv::pow(channels[2], 2, rSq);
    ////沿通道轴求和
    Mat normSqTotal = bSq + gSq + rSq;
    ////计算实际范数（开方）
    Mat norm;
    cv::sqrt(normSqTotal, norm);
    ////元素级除法（向量化实现）
    cv::divide(channels[0], norm+1e-8, channels[0]);  ////除法要避免分母是0的时候
    cv::divide(channels[1], norm+1e-8, channels[1]);
    cv::divide(channels[2], norm+1e-8, channels[2]);
    ////合并通道
    merge(channels, normal_map);
    return normal_map;
}