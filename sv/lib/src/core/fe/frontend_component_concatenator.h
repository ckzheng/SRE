#ifndef FE_RONTEND_COMPONENT_CONCATENATOR_H_
#define FE_RONTEND_COMPONENT_CONCATENATOR_H_
#include "fe/frontend_component.h"

namespace idec {

class FrontendComponent_Concatenator : public FrontendComponentInterface {
 public:

  FrontendComponent_Concatenator(ParseOptions &po,
                                 const std::string name = "Concatenator") : FrontendComponentInterface(po,
                                       name) {
  }

  virtual void Init() {
    FrontendComponentInterface::Init();

    // prepare output-related staff
    output_dim_ = input_dim_;
    output_buff_.Resize(output_dim_, 1);
  }

  virtual bool Process() {
    if (input_buf_.empty())
      return(false);

    // calculate # of frames that can be generated this time
    size_t n = input_buf_[0].NumCols();
    for (size_t i = 1; i < input_buf_.size(); ++i) {
      n = std::min(n, input_buf_[i].NumCols());
    }

    for (size_t t = 0; t < n; ++t) {
      size_t d = 0;
      for (size_t i = 0; i < input_buf_.size(); ++i) {
        xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[i];
#ifdef _MSC_VER
        memcpy_s(output_buff_.Col(0) + d, input_buff.NumRows() * sizeof(float),
                 input_buff.Col(0), input_buff.NumRows() * sizeof(float));
#else
        memcpy(output_buff_.Col(0) + d, input_buff.Col(0),
               input_buff.NumRows() * sizeof(float));
#endif
        d += input_buff.NumRows();
      }

      // push one processed frame to succeeding components, return on error
      if (!SendOneFrameToSucceedingComponents())
        return(false);

      for (size_t i = 0; i < input_buf_.size(); ++i) {
        input_buf_[i].PopfrontOneColumn();
      }
    }

    return(true);
  }
};

}
#endif


