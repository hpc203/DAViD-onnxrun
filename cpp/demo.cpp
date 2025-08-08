#include "depth_estimator.h"
#include "soft_foreground_segmenter.h"
#include "surface_normal_estimator.h"
#include "multi_task_estimator.h"
#include "visualize.h"


using namespace cv;
using namespace std;


Mat display_single_model_results(const Mat& image, std::map<string, Mat>& model_results, const string& prefix)
{
    vector<Mat> visualizations = {image};
    vector<string> labels = {prefix+"/Original"};

    Mat foreground_mask = cv::Mat();
    if(model_results.count("foreground")>0)
    {
        foreground_mask = model_results["foreground"].clone();
    }

    if(model_results.count("depth")>0)
    {
        Mat depth_vis = visualize_relative_depth_map(image, model_results["depth"], foreground_mask, 0);
        visualizations.emplace_back(depth_vis);
        labels.emplace_back(prefix+"/Depth");
    }

    if(model_results.count("foreground")>0)
    {
        Mat foreground_vis = visualize_foreground(image, model_results["foreground"]);
        visualizations.emplace_back(foreground_vis);
        labels.emplace_back(prefix+"/Foreground");
    }
    
    if(model_results.count("normal")>0)
    {
        Mat normal_vis = visualize_normal_maps(image, model_results["normal"], foreground_mask);
        visualizations.emplace_back(normal_vis);
        labels.emplace_back(prefix+"/Normals");
    }

    Mat result = create_concatenated_display(visualizations, labels, 2);
    return result;
}

int main()
{
    string imgpath = "testimgs/1.jpg";
    ////onnx文件路径要写正确
    RelativeDepthEstimator depth_estimator("weights/depth-model-vitb16_384.onnx", true);
    SoftForegroundSegmenter foreground_segmenter("weights/foreground-segmentation-model-vitb16_384.onnx");
    SurfaceNormalEstimator normal_estimator("weights/normal-model-vitb16_384.onnx");
    MultiTaskEstimator multitask_estimator("weights/multi-task-model-vitl16_384.onnx", false);

	Mat image = imread(imgpath);
    if(image.empty())
    {
        cout<<"Error: Could not read the image from"<<imgpath<<", please check your image path"<<endl;
        return -1;
    }
    std::map<string, Mat> individual_results;
    individual_results["depth"] = depth_estimator.estimate_relative_depth(image);
    individual_results["foreground"] = foreground_segmenter.estimate_foreground_segmentation(image);
    individual_results["normal"] = normal_estimator.estimate_normal(image);

    std::map<string, Mat> multitask_results = multitask_estimator.estimate_all_tasks(image);
    
    Mat individual_vis = display_single_model_results(image, individual_results, "Individual");
    Mat multitask_vis = display_single_model_results(image, multitask_results, "Multi-task");
    Mat compare_vis;
    cv::vconcat(individual_vis, multitask_vis, compare_vis);
    
    cv::imwrite("individual_results.jpg", individual_vis);
    cv::imwrite("multitask_results.jpg", multitask_vis);
    cv::imwrite("comparison_results.jpg", compare_vis);
    return 0;
}