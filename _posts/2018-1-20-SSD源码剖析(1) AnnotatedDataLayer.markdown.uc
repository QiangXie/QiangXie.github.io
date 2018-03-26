---
layout: post
title: "SSD源码剖析（1）"
subtitle: "AnnotatedDataLayer层详解"
author: "Johnny"
date: 2018-2-6 11:03:54
header-img: "img/caffe_install.jpg"
tags: 
    - Detection 
---

&#160; &#160; &#160; &#160; AnnotatedDataLayer是SSD代码里一个重要的Layer，它主要用来对数据进行一些数据扩增（Data Augmentation），使得SSD可以使用少量数据进行训练获得较为良好的训练结果。下面是使用官方的SSD进行训练时生成的train net prototxt文件中关于AnnotatedDataLayer的一些参数定义：


    layer {
      name: "data"
      type: "AnnotatedData"
      top: "data"
      top: "label"
      include {
    phase: TRAIN
      }
      transform_param {
    mirror: true
    mean_value: 104.0
    mean_value: 117.0
    mean_value: 123.0
    resize_param {
      prob: 1.0
      resize_mode: WARP
      height: 300
      width: 300
      interp_mode: LINEAR
      interp_mode: AREA
      interp_mode: NEAREST
      interp_mode: CUBIC
      interp_mode: LANCZOS4
    }
    emit_constraint {
      emit_type: CENTER
    }
    distort_param {
      brightness_prob: 0.5
      brightness_delta: 32.0
      contrast_prob: 0.5
      contrast_lower: 0.5
      contrast_upper: 1.5
      hue_prob: 0.5
      hue_delta: 18.0
      saturation_prob: 0.5
      saturation_lower: 0.5
      saturation_upper: 1.5
      random_order_prob: 0.0
    }
    expand_param {
      prob: 0.5
      max_expand_ratio: 4.0
    }
      }
      data_param {
    source: "examples/VOC0712/VOC0712_trainval_lmdb"
    batch_size: 8
    backend: LMDB
      }
      annotated_data_param {
    batch_sampler {
      max_sample: 1
      max_trials: 1
    }
    batch_sampler {
      sampler {
    min_scale: 0.300000011921
    max_scale: 1.0
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
      }
      sample_constraint {
    min_jaccard_overlap: 0.10000000149
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
    min_scale: 0.300000011921
    max_scale: 1.0
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
      }
      sample_constraint {
    min_jaccard_overlap: 0.300000011921
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
    min_scale: 0.300000011921
    max_scale: 1.0
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
      }
      sample_constraint {
    min_jaccard_overlap: 0.5
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
    min_scale: 0.300000011921
    max_scale: 1.0
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
      }
      sample_constraint {
    min_jaccard_overlap: 0.699999988079
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
    min_scale: 0.300000011921
    max_scale: 1.0
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
      }
      sample_constraint {
    min_jaccard_overlap: 0.899999976158
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
    min_scale: 0.300000011921
    max_scale: 1.0
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
      }
      sample_constraint {
    max_jaccard_overlap: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
    label_map_file: "data/VOC0712/labelmap_voc.prototxt"
      }
    }

&#160; &#160; &#160; &#160;下面结合源码来了解这些参数的意义以及SSD如何进行数据增广，下面是SSD中AnnotatedDataLayer的头文件：

    #ifndef CAFFE_DATA_LAYER_HPP_
    #define CAFFE_DATA_LAYER_HPP_
    
    #include <string>
    #include <vector>
    
    #include "caffe/blob.hpp"
    #include "caffe/data_reader.hpp"
    #include "caffe/data_transformer.hpp"
    #include "caffe/internal_thread.hpp"
    #include "caffe/layer.hpp"
    #include "caffe/layers/base_data_layer.hpp"
    #include "caffe/proto/caffe.pb.h"
    #include "caffe/util/db.hpp"
    
    namespace caffe {
    
    template <typename Dtype>
    class AnnotatedDataLayer : public BasePrefetchingDataLayer<Dtype> {
     public:
      explicit AnnotatedDataLayer(const LayerParameter& param);
      virtual ~AnnotatedDataLayer();
      virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      // AnnotatedDataLayer uses DataReader instead for sharing for parallelism
      virtual inline bool ShareInParallel() const { return false; }
      virtual inline const char* type() const { return "AnnotatedData"; }
      virtual inline int ExactNumBottomBlobs() const { return 0; }
      virtual inline int MinTopBlobs() const { return 1; }
    
     protected:
      virtual void load_batch(Batch<Dtype>* batch);
    
      DataReader<AnnotatedDatum> reader_;
      bool has_anno_type_;
      AnnotatedDatum_AnnotationType anno_type_;
      vector<BatchSampler> batch_samplers_;
      string label_map_file_;
    };
    
    }  // namespace caffe
    
    #endif  // CAFFE_DATA_LAYER_HPP_

&#160; &#160; &#160; &#160;AnnotatedDataLayer继承自BasePrefetchingDataLayer，这里面最重要的两个成员函数是    `virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);`和`virtual void load_batch(Batch<Dtype>* batch);`。前者用于对AnnotatedDataLayer进行设置，也可以看作是初始化，后者用于产生进行增广之后在训练中实际用到的数据。

    template <typename Dtype>
    void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      const int batch_size = this->layer_param_.data_param().batch_size();
      const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
      for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    //prototxt 文件里面定义的sampler个数
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
      }
      label_map_file_ = anno_data_param.label_map_file();
      // Make sure dimension is consistent within batch.
      const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
      if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
    ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
    << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
      }
    
      // Read a data point, and use it to initialize the top blob.
      AnnotatedDatum& anno_datum = *(reader_.full().peek());
    
      // Use data_transformer to infer the expected blob shape from anno_datum.
      vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
      this->transformed_data_.Reshape(top_shape);
      // Reshape top[0] and prefetch_data according to the batch_size.
      top_shape[0] = batch_size;
      top[0]->Reshape(top_shape);
      for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
      }
      LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
      // label
      if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
    // If anno_type is provided in AnnotatedDataParameter, replace
    // the type stored in each individual AnnotatedDatum.
    LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
    anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
    // Since the number of bboxes can be different for each image,
    // we store the bbox information in a specific format. In specific:
    // All bboxes are stored in one spatial plane (num and channels are 1)
    // And each row contains one and only one box in the following format:
    // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
    // Note: Refer to caffe.proto for details about group_label and
    // instance_id.
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      num_bboxes += anno_datum.annotation_group(g).annotation_size();
    }
    label_shape[0] = 1;
    label_shape[1] = 1;
    // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
    // cpu_data and gpu_data for consistent prefetch thread. Thus we make
    // sure there is at least one bbox.
    label_shape[2] = std::max(num_bboxes, 1);
    label_shape[3] = 8;
      } else {
    LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
      }
    }
    template<typename Dtype>
    void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
      CPUTimer batch_timer;
      batch_timer.Start();
      double read_time = 0;
      double trans_time = 0;
      CPUTimer timer;
      CHECK(batch->data_.count());
      CHECK(this->transformed_data_.count());
    
      // Reshape according to the first anno_datum of each batch
      // on single input batches allows for inputs of varying dimension.
      const int batch_size = this->layer_param_.data_param().batch_size();
      const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
      const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
      AnnotatedDatum& anno_datum = *(reader_.full().peek());
      // Use data_transformer to infer the expected blob shape from anno_datum.
      vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    
      Dtype* top_data = batch->data_.mutable_cpu_data();
      Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
      if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
      }
    
      // Store transformed annotation.
      map<int, vector<AnnotationGroup> > all_anno;
      int num_bboxes = 0;
    
      for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
    distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
    expand_datum = new AnnotatedDatum();
    this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
    expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
    expand_datum = new AnnotatedDatum();
    this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
    expand_datum = &anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
    // Randomly pick a sampled bbox and crop the expand_datum.
    int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
    sampled_datum = new AnnotatedDatum();
    this->data_transformer_->CropImage(*expand_datum,
       sampled_bboxes[rand_idx],
       sampled_datum);
    has_sampled = true;
      } else {
    sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
    this->data_transformer_->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
      ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
    this->transformed_data_.Reshape(shape);
    batch->data_.Reshape(shape);
    top_data = batch->data_.mutable_cpu_data();
      } else {
    CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
      shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
    shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
    // Make sure all data have same annotation type.
    CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
    if (anno_data_param.has_anno_type()) {
      sampled_datum->set_type(anno_type_);
    } else {
      CHECK_EQ(anno_type_, sampled_datum->type()) <<
      "Different AnnotationType.";
    }
    // Transform datum and annotation_group at the same time
    transformed_anno_vec.clear();
    this->data_transformer_->Transform(*sampled_datum,
       &(this->transformed_data_),
       &transformed_anno_vec);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      // Count the number of bboxes.
      for (int g = 0; g < transformed_anno_vec.size(); ++g) {
    num_bboxes += transformed_anno_vec[g].annotation_size();
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
    all_anno[item_id] = transformed_anno_vec;
      } else {
    	  LOG(INFO) << "has_anno_type_ is false.";
    this->data_transformer_->Transform(sampled_datum->datum(),
       &(this->transformed_data_));
    // Otherwise, store the label from datum.
    CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
    top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
    	LOG(INFO) << "this->ouput_labels_ is false."; 
      this->data_transformer_->Transform(sampled_datum->datum(),
     &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();
    
    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
      }
    
      // Store "rich" annotation if needed.
      if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
    // Store all -1 in the label.
    label_shape[2] = 1;
    batch->label_.Reshape(label_shape);
    caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
    // Reshape the label and store the annotation.
    label_shape[2] = num_bboxes;
    batch->label_.Reshape(label_shape);
    top_label = batch->label_.mutable_cpu_data();
    int idx = 0;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
      for (int g = 0; g < anno_vec.size(); ++g) {
    const AnnotationGroup& anno_group = anno_vec[g];
    for (int a = 0; a < anno_group.annotation_size(); ++a) {
      const Annotation& anno = anno_group.annotation(a);
      const NormalizedBBox& bbox = anno.bbox();
      top_label[idx++] = item_id;
      top_label[idx++] = anno_group.group_label();
      top_label[idx++] = anno.instance_id();
      top_label[idx++] = bbox.xmin();
      top_label[idx++] = bbox.ymin();
      top_label[idx++] = bbox.xmax();
      top_label[idx++] = bbox.ymax();
      top_label[idx++] = bbox.difficult();
    }
      }
    }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
      }
      timer.Stop();
      batch_timer.Stop();
      DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
      DLOG(INFO) << " Read time: " << read_time / 1000 << " ms.";
      DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

&#160; &#160; &#160; &#160;如果在prototxt文件里定义了`distort_param`参数，则在`void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)`里会对图片进行颜色扭曲，`load_batch`调用data_transformer.hpp里的`DistortImage`函数按照prototxt里定义的颜色扭曲参数和概率对数据进行颜色扭曲。同样的，在AnnotatedDataLayer里还按照prototxt里的参数定义对训练数据进行了ExpandImage、CropImage等操作从而增广训练数据。

&#160; &#160; &#160; &#160;其中ExpandImage函数如下：

    template <typename Dtype>
    void DataTransformer<Dtype>::ExpandImage(const cv::Mat& img,
     const float expand_ratio,
     NormalizedBBox* expand_bbox,
     cv::Mat* expand_img) {
      const int img_height = img.rows;
      const int img_width = img.cols;
      const int img_channels = img.channels();
    
      // Get the bbox dimension.
      int height = static_cast<int>(img_height * expand_ratio);
      int width = static_cast<int>(img_width * expand_ratio);
      float h_off, w_off;
      caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
      caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);
      h_off = floor(h_off);
      w_off = floor(w_off);
      expand_bbox->set_xmin(-w_off/img_width);
      expand_bbox->set_ymin(-h_off/img_height);
      expand_bbox->set_xmax((width - w_off)/img_width);
      expand_bbox->set_ymax((height - h_off)/img_height);
    
      expand_img->create(height, width, img.type());
      expand_img->setTo(cv::Scalar(0));
      const bool has_mean_file = param_.has_mean_file();
      const bool has_mean_values = mean_values_.size() > 0;
    
      if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    Dtype* mean = data_mean_.mutable_cpu_data();
    for (int h = 0; h < height; ++h) {
      uchar* ptr = expand_img->ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
    for (int c = 0; c < img_channels; ++c) {
      int blob_index = (c * height + h) * width + w;
      ptr[img_index++] = static_cast<char>(mean[blob_index]);
    }
      }
    }
      }
      if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
    "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
    mean_values_.push_back(mean_values_[0]);
      }
    }
    vector<cv::Mat> channels(img_channels);
    cv::split(*expand_img, channels);
    CHECK_EQ(channels.size(), mean_values_.size());
    for (int c = 0; c < img_channels; ++c) {
      channels[c] = mean_values_[c];
    }
    cv::merge(channels, *expand_img);
      }
    
      cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
      img.copyTo((*expand_img)(bbox_roi));
    }
    
    #endif  // USE_OPENCV

&#160; &#160; &#160; &#160;这个函数的作用是按照prototxt文件里的给出的expand参数随机对图片进行expand操作。具体来说就是：先随机求出膨胀系数，按照这个膨胀系数新建一个空的cv::Mat，然后对这个空的cv::Mat按照prototxt给出的mean_value对其进行填充，然后在这填充后的cv::Mat上随机选取一个和原图大小相同的区域，把原图拷贝到这个区域，用expand后的图片进行后续处理以增加数据集。进行expand之后数据的标注随之改变，所以需要对标注进行修改调整，所以expand最后一步是对标注结果进行TransformAnnotation操作。





**参考资料**


 1. [Recurrent Scale Approximation for Object Detection in CNN-arxiv][1]
 2. [RSA-for-object-detection Matlab 版本 --github][2]
 3. [Octave cp2tform.m源码][3]
 4. [matlab的矩阵左除（A\B）是如何实现的？--知乎][4]
 5. [How to capture frame from RTSP Stream witg FFMPEG Api, OpenCV][5]
 6. [FFmpeg源代码简单分析--雷霄骅][6]

 


  [1]: https://arxiv.org/pdf/1707.09531.pdf
  [2]: https://github.com/sciencefans/RSA-for-object-detection
  [3]: https://sourceforge.net/p/octave/image/ci/default/tree/inst/cp2tform.m#l121
  [4]: https://www.zhihu.com/question/25036509
  [5]: http://hasanaga.info/tag/ffmpeg-avframe-to-opencv-mat/
  [6]: http://blog.csdn.net/leixiaohua1020/article/details/44064715
  [7]: https://github.com/QiangXie/FFmpeg-Decoder-Linux