#include "multi_task_estimator.h"


using namespace cv;
using namespace std;


std::map<string, Mat> MultiTaskEstimator::estimate_all_tasks(const Mat& image)
{
    std::tuple<std::vector<cv::Mat>, MetaData_> result = this->_estimate_dense_map(image);
    std::vector<cv::Mat> outputs = std::get<0>(result);
    MetaData_ metadata = std::get<1>(result);

    std::map<string, Mat> results;

    if(outputs.size()>=3)
    {
        Mat depth_raw = outputs[0];
        Mat normal_raw = outputs[1];
        Mat foreground_raw = outputs[2];

        ////Process depth
        Mat depth_map = composite_model_output_to_image(depth_raw, metadata, cv::INTER_CUBIC);
        if(this->is_inverse_depth)
        {
            depth_map *= -1;
        }
        results["depth"] = depth_map;

        ////Process normals
        Mat normal_map = composite_model_output_to_image(normal_raw, metadata, cv::INTER_CUBIC);
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
        results["normal"] = normal_map;

        ////Process foreground
        Mat foreground_map = composite_model_output_to_image(foreground_raw, metadata, cv::INTER_CUBIC);
        foreground_map.setTo(0, foreground_map<0);
        foreground_map.setTo(1, foreground_map>1);
        if(this->binarization_threshold>0)
        {
            Mat seg = cv::Mat::zeros(foreground_map.rows, foreground_map.cols, CV_32FC1);
            seg.setTo(1, foreground_map>this->binarization_threshold);
            seg.convertTo(seg, CV_8UC1);
            results["foreground"] = seg;
        }
        else
        {
            results["foreground"] = foreground_map;
        }
        return results;
    }

    
    
    return results;
}