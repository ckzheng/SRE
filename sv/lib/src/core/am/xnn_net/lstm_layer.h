#ifndef XNN_NET_LSTM_LAYER_H_
#define XNN_NET_LSTM_LAYER_H_
#include "am/xnn_net/layer_base.h"
namespace idec {

//// LSTM layer, simplified model for NL [guangsheng.bgs]
template<class WMatrix, class BMatrix, class MMatrix, class InputMatrix, class OutputMatrix>
class xnnLSTMLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
  friend class xnnLSTMLayer
    < xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix >
    ;
  friend class xnnLSTMLayer
    < xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix >
    ;

 protected:
  WMatrix Wfw_;        // forward weights applied to the input [W^{fw}(ni); W^{fw}(ig); W^{fw}(fg); W^{fw}(og)]^\top
  WMatrix Rfw_;        // forward weights applied to the recurrent output [R^{fw}(ni); R^{fw}(ig); R^{fw}(fg); R^{fw}(og)]^\top
  WMatrix bfw_;        // forward bias [b^{fw}(ni); b^{fw}(ig); b^{fw}(fg); b^{fw}(og)]
  size_t wstride_;
  size_t nThread_;

 public:
  using xnnLayerBase<InputMatrix, OutputMatrix>::supportBlockEval_;

  xnnLSTMLayer() {
    supportBlockEval_ = false;
  }
  xnnLSTMLayer(const
               xnnLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
               &layer);
  virtual void forwardProp(const InputMatrix &v /*input*/,
                           OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const;

  virtual size_t vDim() const { return Wfw_.NumRows(); }
  virtual size_t uDim() const { return Wfw_.NumCols() / 4; }

  virtual XnnLayerType getLayerType() const { return lstmLayer; }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    Wfw_.Serialize(helper);
    Rfw_.Serialize(helper);
    bfw_.Serialize(helper);
    helper.Serialize(wstride_);
    helper.Serialize(nThread_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    Wfw_.Deserialize(helper);
    Rfw_.Deserialize(helper);
    bfw_.Deserialize(helper);
    helper.Deserialize(wstride_);
    helper.Deserialize(nThread_);
  }
};

template<class WMatrix, class BMatrix, class MMatrix, class InputMatrix, class OutputMatrix>
void xnnLSTMLayer<WMatrix, BMatrix, MMatrix, InputMatrix, OutputMatrix>::forwardProp(
  const InputMatrix &v /*input*/, OutputMatrix &u /*output*/,
  std::vector<void *> &intermediate_states) const {
  MMatrix Ofw_;        // forward activations [O^{fw}(ni); O^{fw}(ig); O^{fw}(fg); O^{fw}(og)]^\top
  MMatrix cfw_;        // forward cell state [c_{raw}]
  MMatrix cfwnl_;        // forward cell state (non-linear)

  // prepare intermediate and output
  Ofw_.Resize(wstride_ * 4, v.NumCols());
  cfw_.Resize(wstride_, 1);
  cfwnl_.Resize(wstride_, 1);
  u.Resize(uDim(), v.NumCols());

  // a lot of views to different matrices
  xnnFloatRuntimeMatrixView ufw(u);
  xnnFloatRuntimeMatrixView ofwCol(Ofw_);

  // Step 1. activation calcuated on non-recurrent input
  Ofw_.ScalePlusMatTMat(0, Wfw_, v);
  Ofw_.Plusv(bfw_);

  for (size_t t = 0; t < v.NumCols(); ++t) {
    ofwCol.ColView(t, 1);

    // Step 2. Output recurrent
    if (t != 0) {
      ufw.ColRowView(t - 1, 1, 0, wstride_);
      ofwCol.ScalePlusMatTMat(1.0f, Rfw_, ufw);
    }

    // Step 3. update cell state
    xnnFloatRuntimeMatrixView ni(Ofw_), gate(Ofw_);

    // update cell state by forget gate
    gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
    gate.HardSigmoid();

    if (t != 0) {
      cfw_.ScalePlusvElemProdv(0, cfw_, gate);
    }

    // update cell state by input gate
    gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
    gate.HardSigmoid();

    ni.ColRowView(t, 1, 0, wstride_); // select node input
    ni.Tanh();

    cfw_.ScalePlusvElemProdv(t == 0 ? 0 : 1.0f, ni, gate);
    cfw_.ApplyFloor(-50);
    cfw_.ApplyCeiling(50);

    // Step 4. non-linear cell state
    cfwnl_ = cfw_;
    cfwnl_.Tanh();

    // Step 5. output by gate
    gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select output gate
    gate.HardSigmoid();

    ufw.ColRowView(t, 1, 0, wstride_);
    ufw.ScalePlusvElemProdv(0, cfwnl_, gate);
  }
}
};



#endif
