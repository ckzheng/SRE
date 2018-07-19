#ifndef FE_RONTEND_COMPONENT_DECIMATE_H
#define FE_RONTEND_COMPONENT_DECIMATE_H
#include "fe/frontend_component.h"

namespace idec {

class FrontendComponent_Decimate : public FrontendComponentInterface {
 protected:
  int decimate_rate_;
  int begin_frame_;
  int input_frm_idx_;

 public:
  FrontendComponent_Decimate(ParseOptions &po,
                             int decimate_rate,
                             const std::string name = "Decimate") :
    FrontendComponentInterface(po, name),
    decimate_rate_(decimate_rate) {
    begin_frame_ = 0;
    po.Register(name_ + "::begin-frame",
                &begin_frame_, "the first frame of decimate");
  }

  virtual bool ReceiveOneFrameFromPrecedingComponent(
    FrontendComponentInterface *from,
    const float *data,
    size_t dim) {
    IDEC_ASSERT(input_map_[from] == 0);

    xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[0];
    bool ret = true;
    ret &= input_buff.PushbackOneColumn(data, dim);

    return(true);
  }

  virtual void Reset() {
    FrontendComponentInterface::Reset();
    input_frm_idx_ = 0;
  }

  virtual void Init() {
    FrontendComponentInterface::Init();
    input_frm_idx_ = 0;
    // prepare output-related staff
    output_dim_ = input_dim_;
    output_buff_.Resize(output_dim_, 1);
  }

  virtual bool Process() {
    if (input_buf_.empty())
      return(false);

    xnnFloatRuntimeMatrixCircularBuffer &inputBuff = input_buf_[0];
    while (static_cast<int>(inputBuff.NumCols()) >= 1) {
      output_buff_.SetZero();
      if ((input_frm_idx_ - begin_frame_) % decimate_rate_ == 0) {
#ifdef _MSC_VER
        memcpy_s(output_buff_.Col(0), sizeof(float) * input_dim_,
                 inputBuff.Col(0), sizeof(float) * input_dim_);
#else
        memcpy(output_buff_.Col(0), inputBuff.Col(0),
               sizeof(float) * input_dim_);
#endif
        // push one processed frame to succeeding
        // components, return on error
        if (!SendOneFrameToSucceedingComponents())
          return(false);
      }
      ++input_frm_idx_;
      inputBuff.PopfrontOneColumn();
    }
    return(true);
  }
};
}  // namespace idec
#endif


