#ifndef _XNN_NET_SOFTMAX_LAYER_H
#define _XNN_NET_SOFTMAX_LAYER_H

#include "am/xnn_net/layer_base.h"
namespace idec {
// Softmax layer [guangsheng.bgs]
template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
class XnnSoftmaxLayer : public XnnLinearLayer < WMatrix, BMatrix, InputMatrix, OutputMatrix > {
  friend class XnnSoftmaxLayer < xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix >;
  friend class XnnSoftmaxLayer < xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix >;

 protected:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::W_;
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::b_;

 public:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::uDim;

  XnnSoftmaxLayer() : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>() {};
  XnnSoftmaxLayer(const XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix> &layer) { *dynamic_cast<XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>*>(this) = layer; };
  XnnSoftmaxLayer(const WMatrix &W, const BMatrix &b) : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>(W, b) {};
  XnnSoftmaxLayer(const XnnSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer);

  virtual XnnLayerType getLayerType() const { return softmaxLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    u.Setv(b_);
    u.PlusMatTMat(W_, v);
    u.Softmax();
  }

  virtual void Serialize(SerializeHelper &helper) {
    XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Serialize(helper);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Deserialize(
      helper);
  }
};

}
#endif
