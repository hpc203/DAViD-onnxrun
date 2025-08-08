#include "visualize.h"


using namespace cv;
using namespace std;


Mat visualize_relative_depth_map(const Mat& frame, const Mat& depth, Mat mask, const float alpha_threshold)
{
    if(!mask.empty())
    {
        Mat processed_depth = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
        Mat foreground;
        cv::bitwise_and(mask > alpha_threshold, depth != 65504, foreground);
        if(cv::countNonZero(foreground) == 0)
        {
            return processed_depth;
        }
        
        vector<Point> non_zero_points;
        cv::findNonZero(foreground, non_zero_points);
        vector<float> values;
        values.reserve(non_zero_points.size());
        for (const auto& p : non_zero_points) {
            values.push_back(depth.at<float>(p.y, p.x));
        }
        const float min_val = *min_element(values.begin(), values.end());
        const float max_val = *max_element(values.begin(), values.end());
        const float denom = (max_val != min_val) ? (max_val - min_val):1e-8;
        const int len_dep = values.size();
        Mat depth_normalized_foreground(len_dep, 1, CV_32FC1);
        for(int i=0;i<len_dep;i++)
        {
            depth_normalized_foreground.at<float>(i, 0) = 1 - ((values[i] - min_val) / denom);
        }
        depth_normalized_foreground *= 255.f;
        depth_normalized_foreground.convertTo(depth_normalized_foreground, CV_8UC1);
        Mat depth_colored_foreground;
        cv::applyColorMap(depth_normalized_foreground, depth_colored_foreground, COLORMAP_INFERNO);
        for(int i=0;i<len_dep;i++)
        {
            const int x = non_zero_points[i].x;
            const int y = non_zero_points[i].y;
            processed_depth.at<Vec3b>(y, x) = depth_colored_foreground.at<Vec3b>(i, 0);
        }
        mask.setTo(0, mask<0);
        mask.setTo(1, mask>1);
        processed_depth.setTo(Scalar(0, 0, 0), mask==0);
        processed_depth.convertTo(processed_depth, CV_32FC3);
        vector<Mat> rgbChannels(3);
        cv::split(processed_depth, rgbChannels);
        vector<Mat> channel_frame(3);
        cv::split(frame, channel_frame);
        vector<Mat> channel_mats(3);
        for(int i=0;i<3;i++)
        {
            channel_frame[i].convertTo(channel_frame[i], CV_32FC1);
            channel_mats[i] = rgbChannels[i].mul(mask) + channel_frame[i].mul(1-mask);
        }
        cv::Mat vis_depth;
        merge(channel_mats, vis_depth);
        vis_depth.convertTo(vis_depth, CV_8UC3);
        return vis_depth;
    }
    else
    {
        double min_val, max_val;
        cv::minMaxLoc(depth, &min_val, &max_val, 0, 0);
        const float denom = (max_val != min_val) ? (max_val - min_val):1e-8;
        Mat depth_normalized = 1 - ((depth - min_val) / denom);
        depth_normalized *= 255.f;
        depth_normalized.convertTo(depth_normalized, CV_8UC1);
        cv::Mat vis_depth;
        cv::applyColorMap(depth_normalized, vis_depth, COLORMAP_INFERNO);
        return vis_depth;
    }

}

Mat visualize_foreground(const Mat& image, Mat mask, const cv::Scalar background_color)
{
    mask.setTo(0, mask<0);
    mask.setTo(1, mask>1);
    mask.convertTo(mask, CV_32FC1);
    vector<Mat> rgb_channel(3);
    cv::split(image, rgb_channel);
    vector<Mat> channel_mats(3);
    for(int i=0;i<3;i++)
    {
        rgb_channel[i].convertTo(rgb_channel[i], CV_32FC1);
        channel_mats[i] = rgb_channel[i].mul(mask) + background_color.val[i] * (1-mask);
    }
    Mat composite_image;
    merge(channel_mats, composite_image);
    composite_image.convertTo(composite_image, CV_8UC3);
    return composite_image;
}

Mat visualize_normal_maps(const Mat& frame, const Mat& normals, Mat mask)
{
    std::vector<Mat> rgb_mats(3);
    cv::split(normals, rgb_mats);
    std::vector<Mat> vis_normals(3);
    vis_normals[2] = (rgb_mats[0] / 2.0 + 0.5) * 255.0;
    vis_normals[1] = (rgb_mats[1] / 2.0 + 0.5) * 255.0;
    vis_normals[0] = (rgb_mats[2] / 2.0 + 0.5) * 255.0;
    if(!mask.empty())
    {
        Mat flag = mask==0;
        mask.setTo(0, mask<0);
        mask.setTo(1, mask>1);
        mask.convertTo(mask, CV_32FC1);
        vector<Mat> rgb_channel(3);
        cv::split(frame, rgb_channel);
        for(int i=0;i<3;i++)
        {
            vis_normals[i].setTo(0, flag);
            rgb_channel[i].convertTo(rgb_channel[i], CV_32FC1);
            vis_normals[i] = vis_normals[i].mul(mask) + rgb_channel[i].mul(1-mask);
        }
    }
    Mat result;
    merge(vis_normals, result);
    result.convertTo(result, CV_8UC3);
    return result;
}

Mat create_concatenated_display(const vector<Mat>& visualizations, const vector<string>& labels, const int downscale)
{
    const int target_height = int(visualizations[0].rows / downscale);
    const int num = visualizations.size();
    vector<Mat> resized_vis;
    for(int i=0;i<num;i++)
    {
        const float aspect_ratio = (float)visualizations[i].cols / visualizations[i].rows;
        const int target_width = int(target_height * aspect_ratio);
        Mat resized;
        cv::resize(visualizations[i], resized, Size(target_width, target_height));
        cv::putText(resized, labels[i], cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        resized_vis.emplace_back(resized);
    }
    
    Mat result;
    cv::hconcat(resized_vis, result);
    return result;
}