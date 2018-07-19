#ifndef FE_RONTEND_COMPONENT_H_
#define FE_RONTEND_COMPONENT_H_


#include <deque>
#include <map>
#include <fstream>
#include <stdlib.h>
#ifndef _MSC_VER
static pthread_mutex_t _RandMutex = PTHREAD_MUTEX_INITIALIZER;
#endif

namespace idec {

const size_t MAX_FRAME_RESERVED = 128;
const size_t FRAME_BLOCK = 8;


class FrontendComponentInterface {
 protected:
  std::string name_;

  std::vector<xnnFloatRuntimeMatrixCircularBuffer> input_buf_;
  std::map<FrontendComponentInterface *, size_t>    input_map_;
  xnnFloatRuntimeMatrix                            output_buff_;
  std::vector<FrontendComponentInterface *>         succeeding_components_;

 public:
  int input_dim_;
  int output_dim_;

 public:

  FrontendComponentInterface(ParseOptions &po, const std::string name) {
    name_ = name;
    input_dim_ = output_dim_ = 0;
  }

  virtual ~FrontendComponentInterface() {
  }

  virtual void Reset() {
    // reset input buffer (call after one utterance ends)
    for (size_t i = 0; i < input_buf_.size(); ++i) {
      input_buf_[i].Clear();
    }
  }

  virtual void Init() {
    // prepare input-related staff
    if (!input_map_.empty()) {
      for (std::map<FrontendComponentInterface *, size_t>::iterator pos =
             input_map_.begin(); pos != input_map_.end(); ++pos) {
        input_buf_[pos->second].Reserve(pos->first->OutputDim(), MAX_FRAME_RESERVED);
        input_dim_ += (int)pos->first->OutputDim();
      }
    } else {
      if (input_dim_ == 0)
        IDEC_ERROR << "input dimension must be set for components with no processors";
      input_map_[(idec::FrontendComponentInterface *)NULL] = 0;
      input_buf_.push_back(xnnFloatRuntimeMatrixCircularBuffer(input_dim_,
                           MAX_FRAME_RESERVED));
    }
  }

  virtual bool Process() {
    if (input_dim_ != output_dim_)
      return(false);

    while (!input_buf_[0].Empty()) {
#ifdef _MSC_VER
      memcpy_s(output_buff_.Col(0), output_dim_ * sizeof(float),
               input_buf_[0].Col(0), input_dim_ * sizeof(float));
#else
      memcpy(output_buff_.Col(0), input_buf_[0].Col(0), input_dim_ * sizeof(float));
#endif
      // push one processed frame to succeeding components, return on error
      if (!SendOneFrameToSucceedingComponents())
        return(false);

      input_buf_[0].PopfrontOneColumn();
    }

    return(true);
  }

  virtual bool Finalize() { return(Process()); };

  bool SendOneFrameToSucceedingComponents(float *data = NULL) {
    if (succeeding_components_.empty())
      return(false);
    if (data == NULL)
      data = output_buff_.Col(0);
    // check if all succeeding components have at least one empty slot
    for (size_t i = 0; i < succeeding_components_.size(); ++i) {
      if (succeeding_components_[i]->NumEmpty(this) == 0)
        return(false);
    }

    // push one frame to all successors
    bool ret = true;
    for (size_t i = 0; i < succeeding_components_.size(); ++i) {
      ret &= succeeding_components_[i]->ReceiveOneFrameFromPrecedingComponent(this,
             data, output_dim_);
    }

    return(ret);
  }


  virtual bool ReceiveOneFrameFromPrecedingComponent(FrontendComponentInterface
      *from, const float *data, size_t dim) {
    return(input_buf_[input_map_[from]].PushbackOneColumn(data, dim));
  }

  void ConnectToPred(FrontendComponentInterface *preceding_component) {
    // connect this to a preceding component
    if (preceding_component != NULL)
      preceding_component->succeeding_components_.push_back(this);

    input_map_.insert(std::make_pair(preceding_component, input_buf_.size()));
    input_buf_.push_back(xnnFloatRuntimeMatrixCircularBuffer());
  }

  size_t InputDim()  { return input_dim_; }
  size_t OutputDim() { return output_dim_; }
  virtual size_t NumEmpty(FrontendComponentInterface *from) { return input_buf_[input_map_[from]].NumEmpty(); }

  std::string GetName() { return name_; }
};
}
#endif


