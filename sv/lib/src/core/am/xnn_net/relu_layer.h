#ifndef XNN_NET_RELU_LAYER_H_
#define XNN_NET_RELU_LAYER_H_
#include "am/xnn_net/linear_layer.h"
namespace idec {

// ReLU layer [zhijie.yzj]
template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
class xnnReLULayer : public XnnLinearLayer < WMatrix, BMatrix, InputMatrix, OutputMatrix > {
 protected:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::W_;
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::b_;

 public:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::uDim;

  xnnReLULayer() : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>() {};
  xnnReLULayer(const XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix> &layer) { *dynamic_cast<XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>*>(this) = layer; };
  xnnReLULayer(const WMatrix &W, const BMatrix &b) : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>(W, b) {};

  virtual XnnLayerType getLayerType() const { return reluLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/,
                           OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    u.Setv(b_);
    u.PlusMatTMat(W_, v);
    u.ReLU();
  }

};
};

#endif
