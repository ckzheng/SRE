#ifndef _XNN_NET_RESCALE_LAYER_H
#define _XNN_NET_RESCALE_LAYER_H

#include "am/xnn_net/layer_base.h"
namespace idec {

// rescale layer [shaofei.xsf]
template<class InputMatrix, class OutputMatrix> class xnnRescaleLayer : public
  xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  OutputMatrix scale_data_;

 public:
  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u = v;
    u.Multiv(scale_data_);
  }

  virtual size_t vDim() const { return scale_data_.NumRows(); }
  virtual size_t uDim() const { return scale_data_.NumRows(); }

  virtual XnnLayerType getLayerType() const { return rescaleLayer; }

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
    scale_data_.Resize(size, 1);
    if (size > 0)
      is.read(reinterpret_cast<char *>(scale_data_.Col(0)), sizeof(Real)*size);
    if (is.fail()) IDEC_ERROR << "read scale_data_ error";

    //for (int i = 0; i < size; i++)
    //    scale_data_.Col(0)[i] = 1.0f / scale_data_.Col(0)[i];
  }

  bool empty() const { return scale_data_.Empty(); }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    scale_data_.Serialize(helper);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    scale_data_.Deserialize(helper);
  }
};
}
#endif
