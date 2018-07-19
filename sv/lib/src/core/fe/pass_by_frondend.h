#ifndef _PASS_BY_FRONTEND_H
#define _PASS_BY_FRONTEND_H
#include <vector>
#include <algorithm>
#include "fe/frontend.h"
#include "am/xnn_runtime.h"
#include "base/log_message.h"
#include "util/parse-options.h"

namespace idec {


class PassByFrondEnd :public FrontEnd {
 public:
  PassByFrondEnd(std::string fe_name = "PassByFrondEnd") {
    dim_ = 0;
    fe_name_ = fe_name;
    data_buf_size_ = 100;
    data_buf_.resize(data_buf_size_);
  }
  ~PassByFrondEnd() {}

  virtual void  Init(const std::string &cfg_file, const std::string sys_dir) {
    ParseOptions po("parses for PassByFrondEnd");
    po.Register(fe_name_ + ".featureDim", &dim_, "featureDim");
    po.Register(fe_name_ + ".sampleRate", &sample_rate_, "featureDim");
    po.ReadConfigFile(cfg_file);
    if (dim_ == 0) {
      IDEC_ERROR << "must set the dim for the" << fe_name_ << "\n";
    }
  };

  virtual int     NumSamplePerFrameShift() {
    IDEC_ERROR << "only support raw feature input for " << fe_name_;
    return 0;
  }
  virtual int     NumSamplePerFrameWindow() {
    IDEC_ERROR << "only support raw feature input for " << fe_name_;
    return 0;
  }
  virtual size_t  FrameShiftInMs() {
    IDEC_ERROR << "only support raw feature input for " << fe_name_;
    return 0;
  }

  // send the beg/end signal to the fe
  virtual bool   BeginUtterance() { total_frm_num_ = 0; frm_num_read_ = 0; return true; }
  virtual bool   EndUtterance() { return true; }            // called at the end of a decoding run, to get set up for another utterance

  // consume audio and output feature
  virtual void    PushAudio(void *buf, int buf_len, IDEC_FE_AUDIOFORMAT format) {

    if (format != FE_RAW_FEATURE_FORMAT) {
      IDEC_ERROR << "only support raw feature input for " << fe_name_;
    }
    if (buf_len % (dim_*sizeof(float)) != 0) {
      IDEC_ERROR <<
                 "only support raw feature input with multiples of the feature frames configure feature dim is:"
                 << dim_;
    }

    // enlarge the space
    size_t frm_num = buf_len / (dim_*sizeof(float));
    if (total_frm_num_ + frm_num > data_buf_size_) {
      data_buf_size_ = std::max(data_buf_size_ * 2, total_frm_num_ + frm_num);
      data_buf_.resize(data_buf_size_);
    }

    // copy in
    for (size_t i = 0; i < frm_num; i++) {
      data_buf_[total_frm_num_ + i].resize(dim_);
      memcpy((void *)(&data_buf_[frm_num_read_ + i][0]),
             (char *)buf + dim_*sizeof(float)*i, dim_*sizeof(float));
    }
    total_frm_num_ += frm_num;
  }

  virtual size_t  NumFrames() { return total_frm_num_ - frm_num_read_; }
  virtual void    LoadKaldiFeatureArk(std::string featFile) {}


  // to be implemented for BLSTM case
  virtual size_t PeekNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    IDEC_ERROR << "not implemented yet";
    return 0;
  }

  virtual size_t  PopNFrames(size_t N) {
    IDEC_ERROR << "not implemented yet";
    return 0;
  }

  virtual size_t  PopNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    if (frm_num_read_ < total_frm_num_) {
      size_t num_remain = total_frm_num_ - frm_num_read_;
      size_t num_read = std::min(num_remain, N);
      ret.Resize(dim_, num_read);

      for (size_t i = 0; i < num_read; ++i) {
        memcpy(ret.Col(i), (void *)(&data_buf_[frm_num_read_ + i][0]),
               dim_*sizeof(float));
      }
      frm_num_read_ += num_read;
      return num_read;
    }
    return 0;
  }

  size_t  total_frm_num_;                          // frame number
  size_t  frm_num_read_;                           // frame number
  std::vector<std::vector<float> >
  data_buf_;      // feature buffer, represented as matrix
  size_t  data_buf_size_;

};

}
#endif
