#ifndef FE_RONTEND_COMPONENT_CONTEXT_EXPANSION_H
#define FE_RONTEND_COMPONENT_CONTEXT_EXPANSION_H
#include "fe/frontend_component.h"

namespace idec {


class FrontendComponent_ContextExpansion : public FrontendComponentInterface {
 protected:
  int minus_, plus_;

 public:

  FrontendComponent_ContextExpansion(ParseOptions &po,
                                     const std::string name = "ContextExpansion") : FrontendComponentInterface(po,
                                           name) {
    // default control parameters
    minus_ = plus_ = 5;

    // get parameters from config
    po.Register(name_ + "::minus", &minus_, "window size of left context");
    po.Register(name_ + "::plus", &plus_, "window size of right context");
  }

  virtual bool ReceiveOneFrameFromPrecedingComponent(FrontendComponentInterface
      *from, const float *data, size_t dim) {
    IDEC_ASSERT(input_map_[from] == 0);

    xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[0];
    bool ret = true;
    if (input_buff.Empty()) {
      // first frame, must have minus_ free space
      if ((int)input_buff.NumEmpty() < minus_ + 1)
        return(false);

      // add the first frame for #window times
      for (int i = 0; i < minus_; ++i) {
        ret &= input_buff.PushbackOneColumn(data, dim);
      }
    }

    if (input_buff.NumEmpty() < 1)
      return(false);
    ret &= input_buff.PushbackOneColumn(data, dim);

    return(true);
  }

  virtual void Init() {
    FrontendComponentInterface::Init();

    // prepare output-related staff
    output_dim_ = input_dim_ * (minus_ + plus_ + 1);
    output_buff_.Resize(output_dim_, 1);

    // check
    if (minus_ < 0 || plus_ < 0)
      IDEC_ERROR << "left- and right-context window size [" << minus_ << ", " <<
                 plus_ << "] must all be non-negative";
  }

  virtual bool Process() {
    if (input_buf_.empty())
      return(false);

    xnnFloatRuntimeMatrixCircularBuffer &inputBuff = input_buf_[0];
    while ((int)inputBuff.NumCols() >= minus_ + plus_ + 1) {
      output_buff_.SetZero();

      for (int t = 0; t < minus_ + plus_ + 1; ++t) {
#ifdef _MSC_VER
        memcpy_s(output_buff_.Col(0) + input_dim_ * t, sizeof(float) * input_dim_,
                 inputBuff.Col(t), sizeof(float) * input_dim_);
#else
        memcpy(output_buff_.Col(0) + input_dim_ * t, inputBuff.Col(t),
               sizeof(float) * input_dim_);
#endif
      }

      // push one processed frame to succeeding components, return on error
      if (!SendOneFrameToSucceedingComponents())
        return(false);

      inputBuff.PopfrontOneColumn();
    }

    return(true);
  }

  virtual bool Finalize() {
    IDEC_ASSERT(!input_buf_.empty());

    xnnFloatRuntimeMatrixCircularBuffer &inputBuff = input_buf_[0];

    bool ret = true;
    if (!inputBuff.Empty()) {
      // push last frame for window_ times
      for (int i = 0; ret && i < plus_; ++i)
        ret &= inputBuff.PushbackOneColumn(inputBuff.Col(inputBuff.NumCols() - 1),
                                           input_dim_);
    }
    return(ret & Process());
  }
};
}
#endif


