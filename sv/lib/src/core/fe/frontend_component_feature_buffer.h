#ifndef FE_RONTEND_COMPONENT_FEATURE_BUF_H_
#define FE_RONTEND_COMPONENT_FEATURE_BUF_H_
#include "fe/frontend_component.h"

namespace idec {


class FrontendComponent_FeatureBuffer : public FrontendComponentInterface {
 public:

  FrontendComponent_FeatureBuffer(ParseOptions &po,
                                  const std::string name = "FeatureBuffer") : FrontendComponentInterface(po,
                                        name) {
  }

  virtual void Init() {
    // prepare input-related staff
    if (input_map_.size() != 1) {
      IDEC_ERROR <<
                 "FrontendComponent_Concatenator must have single preceding component";
    }

    if (input_map_.begin()->first != NULL) {
      input_dim_ = (int)input_map_.begin()->first->OutputDim();

      // prepare output-related staff
      output_dim_ = input_dim_;
    }
    if (input_dim_ == 0)
      IDEC_ERROR << "input dimension not set";
    input_buf_[0].Reserve(input_dim_, MAX_FRAME_RESERVED);
    output_buff_.Resize(output_dim_, 1);
  }

  virtual bool ReceiveOneFrameFromPrecedingComponent(FrontendComponentInterface
      *from, const float *data, size_t dim) {
    // always allocate enough space
    if (input_buf_[input_map_[from]].NumEmpty() == 0) {
      input_buf_[input_map_[from]].Reserve(input_buf_[input_map_[from]].NumRows(),
                                           input_buf_[input_map_[from]].NumCols() * 2);
    }

    return(input_buf_[input_map_[from]].PushbackOneColumn(data, dim));
  }

  size_t PeekNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    return(input_buf_[0].PeekNFrames(N, ret));
  }

  size_t PopNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    return(input_buf_[0].PopNFrames(N, ret));
  }

  /////////////////////zhuozhu.zz ////////////////////
  size_t PeekPartFrames(size_t begin_frm, size_t end_frm,
                        xnnFloatRuntimeMatrix &ret) {
    return(input_buf_[0].PeekPartFrames(begin_frm, end_frm, ret));
  }

  size_t PeekPartFrames(const std::vector<int>&frames_index,
	  xnnFloatRuntimeMatrix &ret) {
	  return(input_buf_[0].PeekPartFrames(frames_index, ret));
  }
  //zhuozhu.zz

  size_t PopNFrames(size_t N) {
    return(input_buf_[0].PopNFrames(N));
  }

  size_t NumFrames() {
    return(input_buf_[0].NumCols());
  }

  bool Empty() {
    return input_buf_[0].Empty();
  }

  virtual size_t NumEmpty(FrontendComponentInterface *from) {
    if (input_buf_[input_map_[from]].NumEmpty() != 0)
      return(input_buf_[input_map_[from]].NumEmpty());

    if (input_buf_[input_map_[from]].NumRows() == 0
        || input_buf_[input_map_[from]].NumCols() == 0)
      return(0);

    // double the capacity
    input_buf_[input_map_[from]].Reserve(input_buf_[input_map_[from]].NumRows(),
                                         input_buf_[input_map_[from]].NumCols() * 2);
    return(input_buf_[input_map_[from]].NumEmpty());
  }

};

}
#endif


