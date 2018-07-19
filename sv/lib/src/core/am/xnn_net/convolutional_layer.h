#ifndef _XNN_NET_CONVOLUTIONAL_LAYER_H
#define _XNN_NET_CONVOLUTIONAL_LAYER_H

#include "am/xnn_net/layer_base.h"
namespace idec {

// convolution layer [shaofei.xsf]
template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
class xnnConvolutionalLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
  friend class xnnConvolutionalLayer < xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix >;
  friend class xnnConvolutionalLayer < xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix >;
 protected:
  WMatrix W_;
  BMatrix b_;
  size_t  vdim_;
  size_t  udim_;
  size_t  patch_dim_;
  size_t  patch_step_;
  size_t  patch_stride_;
 public:
  using xnnLayerBase<InputMatrix, OutputMatrix>::supportBlockEval_;
  xnnConvolutionalLayer() {};
  xnnConvolutionalLayer(const xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer);
  virtual XnnLayerType getLayerType() const { return convolutionalLayer; }
  virtual size_t       vDim() const { return vdim_; }
  virtual size_t       uDim() const { return udim_; }
  virtual void         Serialize(SerializeHelper &helper);
  virtual void         Deserialize(SerializeHelper &helper);
  virtual void         forwardProp(const InputMatrix &v , OutputMatrix &u, std::vector<void *> &intermediate_states) const;

  void readKaldiLayerNnet1(std::istream &is);
  void setvDim(size_t vdim) { vdim_ = vdim; }
  void setuDim(size_t udim) { udim_ = udim; }
};


template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
void xnnConvolutionalLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::readKaldiLayerNnet1(std::istream &is) {
  using namespace xnnKaldiUtility;

  bool binary = true; // always use binary format

  int32 patch_dim;
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<PatchDim>");
    ReadBasicType(is, binary, &patch_dim);
  }
  patch_dim_ = patch_dim;

  int32 patch_step;
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<PatchStep>");
    ReadBasicType(is, binary, &patch_step);
  }
  patch_step_ = patch_step;

  int32 patch_stride;
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<PatchStride>");
    ReadBasicType(is, binary, &patch_stride);
  }
  patch_stride_ = patch_stride;

  int32 learn_rate_coef_;
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<LearnRateCoef>");
    ReadBasicType(is, binary, &learn_rate_coef_);
  }
  int32 bias_learn_rate_coef_;
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<BiasLearnRateCoef>");
    ReadBasicType(is, binary, &bias_learn_rate_coef_);
  }
  int32 max_norm_;
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<MaxNorm>");
    ReadBasicType(is, binary, &max_norm_);
  }

  // filters
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<Filters>");
  }
  int peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed matrix supported";

  std::string token;
  ReadToken(is, binary, &token);
  if (token != "FM") {
    IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;
  }

  int32 rows, cols;
  ReadBasicType(is, binary, &rows);  // throws on error.
  ReadBasicType(is, binary, &cols);  // throws on error.


  W_.Resize(cols, rows);
  // weights
  for (int32 i = 0; i < rows; i++) {
    is.read(reinterpret_cast<char *>(W_.Col(i)), sizeof(Real)*cols);
    if (is.fail()) IDEC_ERROR << "read filters weights";
  }

  // bias
  if ('<' == Peek(is, binary)) {
    ExpectToken(is, binary, "<Bias>");
  }
  peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed vector supported";

  ReadToken(is, binary, &token);
  if (token != "FV")
    IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

  int32 size;
  ReadBasicType(is, binary, &size);  // throws on error.
  b_.Resize(size, 1);
  if (size > 0)
    is.read(reinterpret_cast<char *>(b_.Col(0)), sizeof(Real)*size);
  if (is.fail()) IDEC_ERROR << "read bias error";
}

template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
void xnnConvolutionalLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Serialize(
  SerializeHelper &helper) {
  xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
  W_.Serialize(helper);
  b_.Serialize(helper);
  helper.Serialize(vdim_);
  helper.Serialize(udim_);
  helper.Serialize(patch_dim_);
  helper.Serialize(patch_step_);
  helper.Serialize(patch_stride_);
}

template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
void xnnConvolutionalLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Deserialize(
  SerializeHelper &helper) {
  xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
  W_.Deserialize(helper);
  b_.Deserialize(helper);
  helper.Deserialize(vdim_);
  helper.Deserialize(udim_);
  helper.Deserialize(patch_dim_);
  helper.Deserialize(patch_step_);
  helper.Deserialize(patch_stride_);
}

}
#endif
