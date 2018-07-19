#ifndef XNN_NET_NORM_LAYER_H_
#define XNN_NET_NORM_LAYER_H_

#include "am/xnn_net/layer_base.h"

namespace idec {

// kaldi normalization layer [zhijie.yzj], Matrix for input/output
template<class InputMatrix, class OutputMatrix> class xnnNormLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  size_t dim_;
  mutable OutputMatrix norm_;
  const float kNormFloor;// = (float)pow(2.0, -66);

 public:
  xnnNormLayer() : kNormFloor((float)pow(2.0, -66)) {};
  xnnNormLayer(const size_t dim) : dim_(dim), kNormFloor((float)pow(2.0, -66)) {};

  virtual XnnLayerType getLayerType() const { return normLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    norm_.Resize(v.NumCols(), 1);
    norm_.L2norm(v, kNormFloor);
    u = v;
    u.DivNorm(norm_);
  }

  OutputMatrix getNorm() { return norm_; }
  float getKnormfloor() { return kNormFloor; }

  void setNormLayer(size_t dim, OutputMatrix norm) {
    dim_ = dim;
    norm_ = norm;
    //kNormFloor = knormfloor;
  }

  virtual size_t vDim() const { return dim_; }
  virtual size_t uDim() const { return dim_; }

  virtual void Serialize(SerializeHelper &helper) {
    uint32 dim32 = static_cast<uint32>(dim_);
    helper.Serialize(dim32);
    norm_.Serialize(helper);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    uint32 dim32 = 0;
    helper.Deserialize(dim32);
    dim_ = static_cast<size_t>(dim32);
    norm_.Deserialize(helper);
  }

  void readKaldiLayer(std::istream &is) {
    using namespace xnnKaldiUtility;

    bool binary = true; // always use binary format
    int32 dim;

    std::ostringstream ostr_beg, ostr_end;
    ostr_beg << "<NormalizeComponent>"; // e.g. "<SigmoidComponent>"
    ostr_end << "</NormalizeComponent>"; // e.g. "</SigmoidComponent>"
    ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<Dim>");
    ReadBasicType(is, binary, &dim); // Read dimension
    dim_ = dim;
    std::string tok; // TODO: remove back-compatibility code.
    ReadToken(is, binary, &tok);
    xnnFloatRuntimeMatrix garbage;
    if (tok == "<ValueSum>") {
      //value_sum_.Read(is, binary);
      ReadVector(is, garbage);
      ExpectToken(is, binary, "<DerivSum>");
      //deriv_sum_.Read(is, binary);
      ReadVector(is, garbage);
      ExpectToken(is, binary, "<Count>");
      double count;
      ReadBasicType(is, binary, &count);
      ExpectToken(is, binary, ostr_end.str());
    } else if (tok == "<Counts>") { // Back-compat code for SoftmaxComponent.
      //value_sum_.Read(is, binary); // Set both value_sum_ and deriv_sum_ to the same value,
      ReadVector(is, garbage);
      // and count_ to its sum.
      //count_ = value_sum_.Sum();
      ExpectToken(is, binary, ostr_end.str());
    } else {
      assert(tok == ostr_end.str());
    }
  }
};
};

#endif
