#ifndef XNN_NET_LINERA_LAYER_H_
#define XNN_NET_LINERA_LAYER_H_

#include "am/xnn_net/layer_base.h"

namespace idec {
// Linear layer: y = WX+b [zhijie.yzj]
// WMatrix for weight
// BMatrix for bias
// InputMatrix for input
// OutputMatrix for output
// different Matrix classes for different NN parameters are used because we use quantization, etc.
template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
class XnnLinearLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
  friend class
    XnnLinearLayer< xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix >;
  friend class
    XnnLinearLayer< xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix,  xnnFloat8RuntimeMatrix,  xnnFloatRuntimeMatrix >;
 protected:
  WMatrix W_;
  BMatrix b_;
 public:
  XnnLinearLayer() {};
  XnnLinearLayer(const WMatrix &W, const BMatrix &b) : W_(W), b_(b) {};
  XnnLinearLayer(const
                 XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
                 &layer);

  virtual size_t vDim() const { return W_.NumRows(); }
  virtual size_t uDim() const { return W_.NumCols(); }

  virtual XnnLayerType getLayerType() const { return linearLayer; }
  virtual XnnMatrixType getMatrixType() const;
  virtual void forwardProp(const InputMatrix &v /*input*/,
                           OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    u.Setv(b_);
    u.PlusMatTMat(W_, v);
  }

  void Assign_Wm(std::vector<float> &arr, size_t rows) {
    size_t cols = arr.size() / rows;
    W_.Resize(cols, rows);
    for (size_t i = 0; i < rows; i++) {
      std::vector<float>::iterator it = arr.begin() + i * cols;
      std::copy(it, it + cols, stdext::make_unchecked_array_iterator(W_.Col(i)));
    }
  }

  void Assign_bm(std::vector<float> &arr) {
    b_.Resize(arr.size(), 1);
    std::copy(arr.begin(), arr.end(),
              stdext::make_unchecked_array_iterator(b_.Col(0)));
  }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    W_.Serialize(helper);
    b_.Serialize(helper);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    W_.Deserialize(helper);
    b_.Deserialize(helper);
  }

  void ReadKaldiLayerNnet1(std::istream &is) {
    using namespace xnnKaldiUtility;
    bool binary = true; // always use binary format
    BaseFloat learn_rate_coef_;
    BaseFloat bias_learn_rate_coef_;
    BaseFloat max_norm_;

    //ReadBasicType(is, binary, &rows);
    //ReadBasicType(is, binary, &cols);

    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
      ExpectToken(is, binary, "<BiasLearnRateCoef>");
      ReadBasicType(is, binary, &bias_learn_rate_coef_);
    }

    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<MaxNorm>");
      ReadBasicType(is, binary, &max_norm_);
    }

    // read weight matrix
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
      if (is.fail()) IDEC_ERROR << "read matrix error";
    }

    // read bias
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
    if (is.fail())
      IDEC_ERROR << "read bias error";
  }

  void ReadKaldiLayerNnet2(std::istream &is) {
    using namespace xnnKaldiUtility;

    bool binary = true; // always use binary format

    // read weight matrix
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

    for (int32 i = 0; i < rows; i++) {
      is.read(reinterpret_cast<char *>(W_.Col(i)), sizeof(Real)*cols);
      if (is.fail()) IDEC_ERROR << "read matrix error";
    }

    ExpectToken(is, binary, "<BiasParams>");

    // read bias
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
    if (is.fail())
      IDEC_ERROR << "read bias error";
  }
};
};

#endif
