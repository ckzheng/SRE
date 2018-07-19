#ifndef _XNN_NET_ADD_SHIFT_LAYER_H
#define _XNN_NET_ADD_SHIFT_LAYER_H

#include "am/xnn_net/layer_base.h"
namespace idec {

// addshift layer [shaofei.xsf]
template<class InputMatrix, class OutputMatrix> class xnnAddShiftLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  OutputMatrix shift_data_;

 public:
  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u = v;
    u.Addv(shift_data_);
  }

  virtual size_t vDim() const { return shift_data_.NumRows(); }
  virtual size_t uDim() const { return shift_data_.NumRows(); }

  virtual XnnLayerType getLayerType() const { return addshiftLayer; }

  void readKaldiLayerNnet1(std::istream &is) {
    using namespace xnnKaldiUtility;
    bool binary = true; // always use binary format

    float learn_rate_coef_;
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
    }

    int peekval = is.peek();
    if (peekval != 'F')
      IDEC_ERROR << "Only uncompressed vector supported";

    std::string token;
    ReadToken(is, binary, &token);
    if (token != "FV")
      IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

    int32 size;
    ReadBasicType(is, binary, &size);  // throws on error.
    shift_data_.Resize(size, 1);
    if (size > 0)
      is.read(reinterpret_cast<char *>(shift_data_.Col(0)), sizeof(Real)*size);
    if (is.fail()) IDEC_ERROR << "read scale_data_ error";
  }

  bool empty() const { return shift_data_.Empty(); }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    shift_data_.Serialize(helper);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    shift_data_.Deserialize(helper);
  }
};

}
#endif
