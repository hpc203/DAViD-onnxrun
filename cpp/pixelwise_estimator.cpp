#include "pixelwise_estimator.h"


using namespace cv;
using namespace std;
using namespace Ort;


PixelwiseEstimator::PixelwiseEstimator(string onnxpath)
{
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // 加载模型
#ifdef _WIN32
    std::wstring_convert <std::codecvt_utf8<wchar_t>> converter;
    std::wstring w_onnxModelPath = converter.from_bytes(onnxpath);
    ort_session = new Ort::Session(env, w_onnxModelPath.c_str(), sessionOptions);
#else
    ort_session = new Ort::Session(env, onnxpath.c_str(), sessionOptions);
#endif

    ////高版本onnxruntime在读取输入和输出节点时候的大坑
    Ort::AllocatorWithDefaultOptions allocator;
    // Allocate and store input node names
    auto input_name = ort_session->GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    input_names.push_back(inputNodeNameAllocatedStrings.back().get());

    // Allocate and store output node names
    size_t numOutputNodes = ort_session->GetOutputCount();
    for(size_t i=0;i<numOutputNodes;i++)
    {
        auto output_name = ort_session->GetOutputNameAllocated(i, allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        output_names.push_back(outputNodeNameAllocatedStrings.back().get());
    }
}

std::tuple<std::vector<cv::Mat>, MetaData_> PixelwiseEstimator::_estimate_dense_map(const cv::Mat& image)
{
	////metadata
    MetaData_ metadata;
    Mat processed_image = prepare_image_for_model(image, this->roi_size, metadata);
    
    vector<Mat> rgbChannels(3);
    split(processed_image, rgbChannels);
	const int image_area = processed_image.rows * processed_image.cols;
    this->input_tensor.clear();
	this->input_tensor.resize(1 * 3 * image_area);
    int single_chn_size = image_area * sizeof(float);
	memcpy(this->input_tensor.data(), (float *)rgbChannels[0].data, single_chn_size);
    memcpy(this->input_tensor.data() + image_area, (float *)rgbChannels[1].data, single_chn_size);
    memcpy(this->input_tensor.data() + image_area * 2, (float *)rgbChannels[2].data, single_chn_size);
	
    array<int64_t, 4> input_shape_{ 1, 3, processed_image.rows, processed_image.cols };
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, this->input_tensor.data(), this->input_tensor.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
    std::vector<cv::Mat> maps;
    for(int i=0;i<ort_outputs.size();i++)
    {
        std::vector<int64_t> out_shape = ort_outputs[i].GetTensorTypeAndShapeInfo().GetShape();
        int outHeight = out_shape[1];
        int outWidth = out_shape[2];
        int num_chan = 1;
        if(out_shape.size()==4)
        {
            num_chan = out_shape[1];
            outHeight = out_shape[2];
            outWidth = out_shape[3];
        }
        float* pred = ort_outputs[i].GetTensorMutableData<float>();
        if(num_chan==1)
        {
            Mat res(outHeight, outWidth, CV_32FC1, pred);   ////不考虑batchsize和channel=1
            maps.emplace_back(res);
        }
        else if(num_chan==3)
        {
            const int channel_step = outHeight * outWidth;
            vector<Mat> channel_mats(3);
            channel_mats[0] = cv::Mat(outHeight, outWidth, CV_32FC1, pred);
            channel_mats[1] = cv::Mat(outHeight, outWidth, CV_32FC1, pred+channel_step);
            channel_mats[2] = cv::Mat(outHeight, outWidth, CV_32FC1, pred+2*channel_step);
            Mat res;
            merge(channel_mats, res);
            maps.emplace_back(res);
        }
    }
    std::tuple<std::vector<cv::Mat>, MetaData_> result = std::make_tuple(maps, metadata);
    return result;
}

PixelwiseEstimator::~PixelwiseEstimator()
{
    delete this->ort_session;
    this->ort_session = nullptr;
    std::vector<float>().swap(this->input_tensor);
}


void clip(Mat& m)
{
    m.setTo(0, m<0);
    m.setTo(1, m>1);
}

Mat prepare_image_for_model(const Mat& image, const int roi_size, MetaData_& metadata)
{
    ////preprocess_img
    Mat image_bgr;
    image.convertTo(image_bgr, CV_32FC3, 1 / 255.f);
    vector<Mat> rgbChannels(3);
    split(image_bgr, rgbChannels);
    clip(rgbChannels[0]);
    clip(rgbChannels[1]);
    clip(rgbChannels[2]);
    merge(rgbChannels, image_bgr);

    ////prepare_image_for_model
    metadata.original_shape[0] = image_bgr.rows;
    metadata.original_shape[1] = image_bgr.cols;
    int pad_h = 0;
    int pad_w = 0;
    int pad_h_extra = 0;
    int pad_w_extra = 0;
    const int* original_shape = metadata.original_shape;
    if(original_shape[0] < original_shape[1])
    {
        pad_h = int((original_shape[1] - original_shape[0]) / 2);
        pad_w = 0;
        pad_h_extra = original_shape[1] - original_shape[0] - pad_h;
        pad_w_extra = 0;
    } 
    else if(original_shape[0] > original_shape[1])
    {
        pad_w = int((original_shape[0] - original_shape[1]) / 2);
        pad_h = 0;
        pad_w_extra = original_shape[0] - original_shape[1] - pad_w;
        pad_h_extra = 0;
    }
        
    Mat padded_image;
    cv::copyMakeBorder(image_bgr, padded_image, pad_h, pad_h_extra, pad_w, pad_w_extra, cv::BORDER_REPLICATE);
    metadata.square_shape[0] = padded_image.rows;
    metadata.square_shape[1] = padded_image.cols;

    if(padded_image.cols > roi_size * 3 and padded_image.rows > roi_size * 3)
    {
        cv::pyrDown(padded_image, padded_image);
    }

    Mat resized_image;
    cv::resize(padded_image, resized_image, cv::Size(roi_size, roi_size), cv::INTER_LINEAR);
    metadata.original_padding[0] = pad_h;
    metadata.original_padding[1] = pad_w;
    metadata.original_padding[2] = pad_h_extra;
    metadata.original_padding[3] = pad_w_extra;
    return resized_image;
}

Mat composite_model_output_to_image(const Mat& model_output, const MetaData_& metadata, int interp_mode)
{
    const int pad_h = metadata.original_padding[0];
    const int pad_w = metadata.original_padding[1];
    const int pad_h_extra = metadata.original_padding[2];
    const int pad_w_extra = metadata.original_padding[3];
    Mat resized_to_square;
    cv::resize(model_output, resized_to_square, cv::Size(metadata.square_shape[1], metadata.square_shape[0]), interp_mode);

    Mat final_output;
    if(pad_h > 0 || pad_h_extra > 0)
    {
        resized_to_square.rowRange(pad_h, metadata.square_shape[0] - pad_h_extra).copyTo(final_output);
    }  
    else if(pad_w > 0 || pad_w_extra > 0)
    {
        resized_to_square.colRange(pad_w, metadata.square_shape[1] - pad_w_extra).copyTo(final_output);
    } 
    else
    {
        final_output = resized_to_square.clone();
    }
    return final_output;      
}