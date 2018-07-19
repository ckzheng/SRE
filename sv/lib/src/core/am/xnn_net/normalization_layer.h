#ifndef _XNN_NET_NORMALIZTION_LAYER_H
#define _XNN_NET_NORMALIZTION_LAYER_H
#include "am/xnn_net/layer_base.h"

namespace idec {

// normalization layer, e.g., perform mean and std. var. normalization for the input
template<class InputMatrix, class OutputMatrix> class xnnNormalizationLayer :
  public xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  OutputMatrix       mean_;
  OutputMatrix       stdvar_;
  std::vector<int32> splice_;

 public:
  virtual void forwardProp(const InputMatrix &v /*input*/,
                           OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u = v;
    u.MinusvDivv(mean_, stdvar_);
  }

  virtual size_t vDim() const { return mean_.NumRows(); }
  virtual size_t uDim() const { return mean_.NumRows(); }

  virtual XnnLayerType getLayerType() const { return normalizationLayer; }

  bool empty() const { return mean_.Empty(); }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    mean_.Serialize(helper);
    stdvar_.Serialize(helper);
    helper.Serialize(splice_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    mean_.Deserialize(helper);
    stdvar_.Deserialize(helper);
    helper.Deserialize(splice_);
  }

  void readKaldiLayerNnet1(std::istream &is) {
    using namespace xnnKaldiUtility;

    bool binary = false;

    int32 skip;
    std::vector<BaseFloat> mean;
    std::vector<BaseFloat> stdvar;
    BaseFloat data;
    std::string token;

    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<Splice>");
      ReadBasicType(is, binary, &skip);
      ReadBasicType(is, binary, &skip);
      while (isspace(is.peek())) {
        is.get();  // consume the space.
      }

      if ('[' == is.get()) {
        while (is.peek() != ']') {
          is >> data;
          splice_.push_back((int32)data);
          is.get();
        }
      }
    }

    is.get();// consume the ]
    while (isspace(is.peek())) {
      is.get();  // consume the space.
    }

    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<AddShift>");
      ReadBasicType(is, binary, &skip);
      ReadBasicType(is, binary, &skip);
      while (isspace(is.peek())) {
        is.get();  // consume the space.
      }

      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &skip);
      while (isspace(is.peek())) {
        is.get();  // consume the space.
      }

      if ('[' == is.get()) {
        while (is.peek() != ']') {
          is >> data;
          mean.push_back(data);
          is.get();
        }
      }
    }

    is.get();// consume the ]
    while (isspace(is.peek())) {
      is.get();  // consume the space.
    }

    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<Rescale>");
      ReadBasicType(is, binary, &skip);
      ReadBasicType(is, binary, &skip);
      while (isspace(is.peek())) {
        is.get();  // consume the space.
      }

      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &skip);
      while (isspace(is.peek())) {
        is.get();  // consume the space.
      }

      if ('[' == is.get()) {
        while (is.peek() != ']') {
          is >> data;
          stdvar.push_back(data);
          is.get();
        }
      }
    }

    is.get();// consume the ]
    while (isspace(is.peek())) {
      is.get();  // consume the space.
    }
    ReadToken(is, binary, &token);
    if (token != "</Nnet>") {
      IDEC_ERROR << ": Expected token " << "/Nnet" << ", got " << token;
    }

    mean_.Resize(mean.size(), 1);
    stdvar_.Resize(stdvar.size(), 1);
    for (size_t i = 0; i < mean.size(); i++) {
      mean_.Col(0)[i] = 0.0f - mean[i];
      stdvar_.Col(0)[i] = 1.0f / stdvar[i];
    }
  }
};
}
#endif
