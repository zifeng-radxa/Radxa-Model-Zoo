//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV8_H
#define YOLOV8_H

#include <iostream>
#include <vector>
#include "bm_wrapper.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#include "bmnn_utils.h"


// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0
typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;
typedef double* BB;
typedef struct {
    siz h, w, m;
    uint* cnts;
} RLE;

/* Initialize/destroy RLE. */
void rleInit(RLE* R, siz h, siz w, siz m, uint* cnts);

void rleEncode(RLE* R, const byte* mask, siz h, siz w, siz n);

struct YoloV8Box {
    float x1, y1, x2, y2;
    // int x, y, width, height;
    float score;
    int class_id;
    std::vector<float> mask;
    cv::Mat mask_img;  // 预测出来不仅有框，还有mask
    int temp_id; // 临时记录box在类别中的顺序，略去过滤前的mask填充
};

struct ImageInfo {
    cv::Size raw_size;
    cv::Vec4d trans;
};

struct Paras {
    int r_x;
    int r_y;
    int r_w;
    int r_h;
    int width;
    int height;
};

using YoloV8BoxVec = std::vector<YoloV8Box>;

class YoloV8 {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;

    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    
    // configuration
    float m_confThreshold = 0.25;
    float m_nmsThreshold = 0.7;
    
    std::vector<std::string> m_class_names;
    int m_class_num = 80;  // default is coco names
    int mask_len = 32;
    int m_net_h, m_net_w;
    int max_batch;
    int output_num;
    int min_dim;
    int max_det = 300;
    int max_wh = 7680;  // (pixels) maximum box width and height
    bmcv_convert_to_attr converto_attr;

    
    TimeStamp* m_ts;

   private:
    int pre_process(const std::vector<bm_image>& images);
    int post_process(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& boxes);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    static bool YoloV8Box_cmp(YoloV8Box a, YoloV8Box b);
    void NMS(YoloV8BoxVec& dets, float nmsConfidence);
    void clip_boxes(YoloV8BoxVec& yolobox_vec, int src_w, int src_h);
      
   public:
    YoloV8(std::shared_ptr<BMNNContext> context);
    virtual ~YoloV8();
    int Init(float confThresh = 0.5, float nmsThresh = 0.5, const std::string& coco_names_file = "");
    void enableProfile(TimeStamp* ts);
    int batch_size();
    int Detect(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& boxes);
    void draw_bmcv(bm_handle_t& handle, bm_image& frame, YoloV8BoxVec& result, bool put_text_flag);
    void draw_result(cv::Mat& img, YoloV8BoxVec& result);

    // tpumask model bmrt
   private:
    bm_handle_t tpu_mask_handle;
    void *bmrt = nullptr;                    
    const bm_net_info_t *netinfo;            
    std::vector<std::string> network_names;
    bool tpu_post = false;
    int tpu_mask_num = 32;
    int m_tpumask_net_h, m_tpumask_net_w;

    public:
    void tpumask_Init(std::string bmodel_file, int dev_id = 0);
    void getmask_tpu(YoloV8BoxVec &yolobox_vec, int start, const bm_tensor_t& input_tensor1, Paras& paras);

};

#endif  //! YOLOV8_H
