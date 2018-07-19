#ifndef _MFCC_H_
#define _MFCC_H_

#include <string>
#include <vector>
#include "seg.h"
#include "als_error.h"
#include "new_matrix.h"
#include "fe/frontend.h"
#include "fe/frontend_pipeline.h"

using std::vector;
using std::string;

namespace alspkdiar {

class Mfcc {
 public:
  Mfcc(const char *sys_dir, const char *cfg_file, bool verbose_mode = false,
       std::string str_input_type = "FE_8K_16BIT_PCM") {
    feature_ = NULL;
    front_end_new_ = NULL;
    front_end_ = new idec::FrontendPipeline();
    front_end_->Init(cfg_file, sys_dir);
    verbose_mode_ = verbose_mode;
    str_input_type_ = str_input_type;
  }

  void Init(const char *sys_dir, const char *cfg_file) {
    front_end_new_ = new idec::FrontendPipeline();
    front_end_new_->Init(cfg_file, sys_dir);
  }

  void FENew(char *wave, unsigned int len) {
    // do vad, get the frame_by_frame result
    front_end_new_->BeginUtterance();
    const size_t wavblocksize = 400;
    unsigned int loop = len / wavblocksize;
    if ((len % wavblocksize) != 0) {
      loop += 1;
    }

    unsigned int base_ptr = 0, frag_len = 0;
    for (int i = 0; i < loop; ++i) {
      frag_len = (i == loop - 1) ? (len % wavblocksize) : wavblocksize;
      if (frag_len == 0) {
        frag_len = wavblocksize;
      }

      front_end_new_->PushAudio(wave + base_ptr, frag_len, enum_input_type_);

      base_ptr += frag_len;
    }
    front_end_new_->EndUtterance();
  }

  void XnnMatrix2DoubleMatrix(const idec::xnnFloatRuntimeMatrix &xnnfeat,
                              DoubleMatrix &feats, int idx, int begin_idx, int end_idx) const {
    idec::IDEC_ASSERT(end_idx - begin_idx > 0);
    idec::IDEC_ASSERT(feats.Rows() >= idx + end_idx - begin_idx);
    //feats.Resize(end_idx - begin_idx, xnnfeat.NumRows());
    float *pTmp = NULL;
    for (int i = begin_idx, k = idx; i < end_idx; ++i, ++k) {
      pTmp = xnnfeat.Col(i);
      for (int j = 0; j < xnnfeat.NumRows(); ++j) {
        feats(k, j) = pTmp[j];
      }
    }
  }

  int MergeFeatureNew(const SegCluster &cluster, DoubleMatrix &feats) const {
    int num_frm = 0;
    int total_len = cluster.Length();
    feats.Resize(total_len, front_end_new_->GetFeatureDim());
    for (int i = 0; i < cluster.Size(); ++i) {
      const Seg &segment = cluster.GetSeg(i);
      idec::xnnFloatRuntimeMatrix xnnfeat;
      front_end_new_->PeekPartFrames(segment.begin, segment.end, xnnfeat);
      XnnMatrix2DoubleMatrix(xnnfeat, feats, num_frm, 0, xnnfeat.NumCols());
      num_frm += xnnfeat.NumCols();
    }
    idec::IDEC_ASSERT(total_len == num_frm);
    return 0;
  }

  int MergeFeature(const SegCluster &cluster, DoubleMatrix &feats) const {
    const int len = cluster.Length();
    if (len == 0) {
      return -1;
    }

    feats.Resize(len, dims_);

    int feat_count = 0;
    float **feature = GetFeature();
    const int cluster_size = cluster.Size();
    for (int i = 0; i < cluster_size; ++i) {
      const Seg &seg = cluster.GetSeg(i);
      for (int j = seg.begin; j < seg.end; ++j) {
        idec::IDEC_ASSERT(j < frames_);
        for (int k = 0; k < dims_; ++k) {
          feats(feat_count, k) = feature[j][k];
        }
        ++feat_count;
      }
    }
    return 0;
  }

  int MergeFeature(const SegCluster &cluster, vector<float *> &feats) const {
    const int len = cluster.Length();
    if (len == 0) {
      return -1;
    }

    feats.clear();
    feats.reserve(len);
    float **feature = GetFeature();
    const int cluster_size = cluster.Size();
    for (int i = 0; i < cluster_size; ++i) {
      const Seg &seg = cluster.GetSeg(i);
      for (int j = seg.begin; j < seg.end; ++j) {
        idec::IDEC_ASSERT(j < frames_);
        feats.push_back(feature_[j]);
      }
    }
    return 0;
  }

  void LoadMfcc(const char *path, DoubleMatrix &feats) {
    ifstream ifs(path);
    if (!ifs) {
      idec::IDEC_ERROR << "fail to open file " << path;
    }

    feats.Resize(1427, 60);
    int frames = 0;
    string wave_name;
    string begin_separator;
    string end_separator;
    ifs >> wave_name;
    ifs >> begin_separator;
    if (begin_separator == "[") {
      cout << "read " << path << " begin.." << endl;
    }
    float value;
    int count = 0;
    while (true) {
      ifs >> value;
      //feature_.push_back(value);
      feats(frames, count) = value;
      count++;
      if (count == 60) {
        ++frames;
        // cout << "read line " << frames_ << endl;
        if (frames == 1427) {
          break;
        }
        count = 0;
      }
    }
  }

  ~Mfcc() {
    idec::IDEC_DELETE(front_end_);
    idec::IDEC_DELETE(front_end_new_);
    Clear();
  }

  void Clear() {
    if (feature_ != NULL) {
      for (unsigned int i = 0; i < frames_; ++i) {
        idec::IDEC_DELETE_ARRAY(feature_[i]);
      }
      idec::IDEC_DELETE_ARRAY(feature_);
    }
  }

  unsigned int GetDims() const {
    return dims_;
  }

  unsigned int GetFrames() const {
    return frames_;
  }

  float   **GetFeature() const {
    return feature_;
  }

  int FE(char *wave, unsigned int len) {
    InputWavTypeStr2Enum(str_input_type_, enum_input_type_);
    front_end_->BeginUtterance();
    //front_end_->PushAudio(wave, len, enum_input_type_);
    const size_t wavblocksize = 400;
    unsigned int loop = len / wavblocksize;
    if ((len % wavblocksize) != 0) {
      loop += 1;
    }

    unsigned int base_ptr = 0, begin_pos = 0, frag_len = 0;
    for (int i = 0; i < loop; ++i) {
      frag_len = (i == loop - 1) ? (len % wavblocksize) : wavblocksize;
      if (frag_len == 0) {
        frag_len = wavblocksize;
      }

      front_end_->PushAudio(wave + base_ptr, frag_len, enum_input_type_);

      base_ptr += frag_len;
    }
    front_end_->EndUtterance();

    // get the feature_ in batch mode
    unsigned int num_frm = front_end_->NumFrames();
    this->frames_ = num_frm;
    this->dims_ = front_end_->GetFeatureDim();

    if (num_frm > 0) {
      idec::xnnFloatRuntimeMatrix feat_buf;
      feature_ = new float*[num_frm];
      for (unsigned int i = 0; i < num_frm; ++i) {
        feature_[i] = new float[this->dims_];
      }

      front_end_->PopNFrames(num_frm, feat_buf);

      float *pTmp = NULL;
      float *means = new float[this->dims_];
      float *vars = new float[this->dims_];
      for (unsigned int i = 0; i < this->dims_; ++i) {
        means[i] = 0;
        vars[i] = 0;
      }

      for (unsigned int i = 0; i < num_frm; ++i) {
        pTmp = feat_buf.Col(i);
        for (unsigned int j = 0; j < this->dims_; ++j) {
          feature_[i][j] = pTmp[j];
          means[j] += pTmp[j];
        }
      }

      for (unsigned int i = 0; i < this->dims_; ++i) {
        means[i] /= num_frm;
      }

      for (unsigned int i = 0; i < num_frm; ++i) {
        pTmp = feat_buf.Col(i);
        for (unsigned int j = 0; j < this->dims_; ++j) {
          vars[j] += (pTmp[j] - means[j]) *  (pTmp[j] - means[j]);
        }
      }

      for (unsigned int i = 0; i < this->dims_; ++i) {
        vars[i] /= num_frm - 1;
      }

      for (unsigned int i = 0; i < num_frm; ++i) {
        for (unsigned int j = 0; j < this->dims_; ++j) {
          feature_[i][j] = (feature_[i][j] - means[j]) / sqrt(vars[j]);
        }
      }

      if (verbose_mode_) {
        idec::IDEC_INFO << "total frame number:" << num_frm;
      }

      front_end_->Empty();
      idec::IDEC_DELETE_ARRAY(means);
      idec::IDEC_DELETE_ARRAY(vars);
    }
    return ALS_OK;
  }

  int InputWavTypeStr2Enum(const std::string str_input_str,
                                   idec::IDEC_FE_AUDIOFORMAT &enum_wav_type) {
    enum_wav_type = idec::FE_8K_16BIT_PCM;
    if (str_input_str == "FE_8K_16BIT_PCM") {
      enum_wav_type = idec::FE_8K_16BIT_PCM;
    } else if (str_input_str == "FE_16K_16BIT_PCM") {
      enum_wav_type = idec::FE_16K_16BIT_PCM;
    } else {
      idec::IDEC_ERROR << "unknown input type " << str_input_type_;
    }
    return ALS_OK;
  }

 private:
  bool verbose_mode_;
  std::string str_input_type_;
  idec::FrontendPipeline *front_end_;
  idec::FrontendPipeline *front_end_new_;
  float **feature_;
  unsigned int dims_;
  unsigned int frames_;
  enum idec::IDEC_FE_AUDIOFORMAT enum_input_type_;
};
}
#endif
