project(test)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/pixelwise_estimator.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/depth_estimator.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/soft_foreground_segmenter.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/surface_normal_estimator.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/multi_task_estimator.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/visualize.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/include/opencv4"
        PUBLIC "/home/wangbo/libs/mylibs/onnxruntime-linux-x64-1.18.1/include")

target_link_libraries(${PROJECT_NAME}
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgcodecs.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_highgui.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgproc.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_core.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_videoio.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_video.so.4.8.0"
        "/home/wangbo/libs/mylibs/onnxruntime-linux-x64-1.18.1/lib/libonnxruntime.so.1.18.1")