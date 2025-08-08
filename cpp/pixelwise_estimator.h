# ifndef Pixelwise_Estimator_H
# define Pixelwise_Estimator_H
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>


typedef struct
{
	int original_shape[2];
    int square_shape[2];
    int original_padding[4];
} MetaData_;


class PixelwiseEstimator
{
public:
	PixelwiseEstimator(std::string onnxpath);
	~PixelwiseEstimator();
protected:
	std::tuple<std::vector<cv::Mat>, MetaData_> _estimate_dense_map(const cv::Mat& image);
private:
	std::vector<float> input_tensor;

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Pixelwise Estimator");
	Ort::Session *ort_session = nullptr;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	
	std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> input_names;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> output_names;
    const int roi_size = 512;
};


cv::Mat prepare_image_for_model(const cv::Mat& image, const int roi_size, MetaData_& metadata);
cv::Mat composite_model_output_to_image(const cv::Mat& model_output, const MetaData_& metadata, int interp_mode);


#endif