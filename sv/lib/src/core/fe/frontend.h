#ifndef ASR_DECODER_SRC_CORE_FE_FRONTEND_H_
#define ASR_DECODER_SRC_CORE_FE_FRONTEND_H_

#include <string>
namespace idec {
enum IDEC_FE_AUDIOFORMAT {
  FE_NULL_AUDIOFORMAT = 0,
  FE_8K_16BIT_PCM,
  FE_16K_16BIT_PCM,
  FE_RAW_FEATURE_FORMAT = 9999
};

class xnnFloatRuntimeMatrix;
class FrontEnd {
 protected:
  int dim_;
  int sample_rate_;
  std::string fe_name_;

 public:
  FrontEnd() { dim_ = -1; sample_rate_ = -1; }
  virtual ~FrontEnd() {}

  // build the frontend from configuration file
  virtual void Init(const std::string &cfg_fn, const std::string sys_dir) = 0;

  virtual bool   BeginUtterance() = 0;
  virtual bool   EndUtterance() = 0;

  virtual void   PushAudio(void *buf, int len, IDEC_FE_AUDIOFORMAT format) = 0;
  virtual size_t PopNFrames(size_t N, xnnFloatRuntimeMatrix &ret) = 0;
  virtual size_t PopNFrames(size_t N) = 0;
  virtual size_t PeekNFrames(size_t N, xnnFloatRuntimeMatrix &ret) = 0;

  virtual size_t NumFrames() = 0;  // returns # of frames available
  virtual int    NumSamplePerFrameShift() = 0;
  virtual int    NumSamplePerFrameWindow() = 0;
  virtual size_t FrameShiftInMs() = 0;

  // basic feature information
  int  GetFeatureDim() const { return dim_; }
  void SetFeatureDim(int dim) { dim_ = dim; }
  int  GetSampleRate() const { return sample_rate_; }

  // directly load kaldi feature ark (mainly for debug purpose)
  virtual void LoadKaldiFeatureArk(std::string feat_file) = 0;
};
}  //  namespace idec

#endif  //  ASR_DECODER_SRC_CORE_FE_FRONTEND_H_
