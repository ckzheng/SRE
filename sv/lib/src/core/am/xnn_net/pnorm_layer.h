#ifndef XNN_NET_PNORM_LAYER_H_
#define XNN_NET_PNORM_LAYER_H_

#include "am/xnn_net/layer_base.h"

namespace idec {

// kaldi p-norm layer [zhijie.yzj], Matrix for input/output
template<class InputMatrix, class OutputMatrix>
class xnnPnormLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  size_t vdim_;
  size_t udim_;
  size_t groupsize_;
  float p_;

 public:
  xnnPnormLayer() {};
  xnnPnormLayer(const size_t vdim, const size_t udim, const float p) : vdim_(vdim), udim_(udim), p_(p) {};

  virtual XnnLayerType getLayerType() const { return pnormLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    if (p_ == 2.0f) {
      u.GroupPnorm2(v, groupsize_);
    } else {
      IDEC_ERROR << "not implemented yet";
    }
  }

  void setPnormLayer(size_t u, size_t v, size_t g, float p) {
    udim_ = u;
    vdim_ = v;
    groupsize_ = g;
    p_ = p;
  }

  float getP() { return p_; }
  size_t getGroupsize() { return groupsize_; }

  virtual size_t vDim() const { return vdim_; }
  virtual size_t uDim() const { return udim_; }


  virtual void Serialize(SerializeHelper &helper) {
    helper.Serialize(vdim_);
    helper.Serialize(udim_);
    helper.Serialize(groupsize_);
    helper.Serialize(p_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    helper.Deserialize(vdim_);
    helper.Deserialize(udim_);
    helper.Deserialize(groupsize_);
    helper.Deserialize(p_);
  }


  void readKaldiLayer(std::istream &is) {
    using namespace xnnKaldiUtility;

    bool binary = true; // always use binary format
    int32 vdim, udim;
    Real p;

    ExpectOneOrTwoTokens(is, binary, "<PnormComponent>", "<InputDim>");
    ReadBasicType(is, binary, &vdim);
    ExpectToken(is, binary, "<OutputDim>");
    ReadBasicType(is, binary, &udim);
    ExpectToken(is, binary, "<P>");
    ReadBasicType(is, binary, &p);
    ExpectToken(is, binary, "</PnormComponent>");

    vdim_ = vdim;
    udim_ = udim;
    p_ = p;

    if (p != 0 && p_ != 1.0f && p != 2.0f)
      IDEC_ERROR << "only support p = 0 / 1.0 / 2.0";

    if (vdim_ % udim_ != 0)
      IDEC_ERROR << "vdim and udim mismatch " << vdim_ << " vs. " << udim_;

    groupsize_ = vdim_ / udim_;
  }

};
};

#endif
