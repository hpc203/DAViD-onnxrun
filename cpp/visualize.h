# ifndef Visualize_H
# define Visualize_H
#include <opencv2/imgproc.hpp>


cv::Mat visualize_relative_depth_map(const cv::Mat& frame, const cv::Mat& depth, cv::Mat mask, const float alpha_threshold=0);
cv::Mat visualize_foreground(const cv::Mat& image, cv::Mat mask, const cv::Scalar background_color=cv::Scalar(0, 255, 0));
cv::Mat visualize_normal_maps(const cv::Mat& frame, const cv::Mat& normals, cv::Mat mask);
cv::Mat create_concatenated_display(const std::vector<cv::Mat>& visualizations, const std::vector<std::string>& labels, const int downscale=1);


#endif