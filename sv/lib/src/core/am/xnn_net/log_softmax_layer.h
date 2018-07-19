#ifndef _XNN_NET_LOG_SOFTMAX_LAYER_H
#define _XNN_NET_LOG_SOFTMAX_LAYER_H

namespace idec {

// Log-Softmax layer [zhijie.yzj]
template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
class xnnLogSoftmaxLayer : public XnnLinearLayer < WMatrix, BMatrix, InputMatrix, OutputMatrix > {
  friend class xnnLogSoftmaxLayer < xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix >;
  friend class xnnLogSoftmaxLayer < xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix >;

 protected:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::W_;
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::b_;
  OutputMatrix prior_; // prior
  bool use_real_prob_; // normalize \sum \exp(st), default to false for better performance
  bool use_prior_;     // subtract output by log-prior, default = true

 public:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::uDim;
  xnnLogSoftmaxLayer() : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>() { use_real_prob_ = false; use_prior_ = true; };
  xnnLogSoftmaxLayer(const XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix> &layer) { *dynamic_cast<XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>*>(this) = layer; use_real_prob_ = false; use_prior_ = true; };
  xnnLogSoftmaxLayer(const WMatrix &W, const BMatrix &b) : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>(W, b) { use_real_prob_ = false; use_prior_ = true; };
  xnnLogSoftmaxLayer(const xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer);

  OutputMatrix getPrior_() { return prior_; }
  void setPrior(OutputMatrix prior) { prior_ = prior; }
  bool getUseRealProb() const { return use_real_prob_; }
  bool getUsePrior() const { return use_prior_; }
  bool hasPrior() const { return use_prior_  && !prior_.Empty(); }
  virtual XnnLayerType getLayerType() const { return logsoftmaxLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    u.Setv(b_);
    u.PlusMatTMat(W_, v);

    if (use_real_prob_) {
      u.LogSoftmax();
    }

    if (hasPrior()) {
      u.Minusv(prior_);
    }
  }

  virtual void forwardPropRange(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const;

  void readKaldiPu(std::istream &is);
  void readKaldiNnet1Pu(std::istream &is);

  void useRealProb(bool use_real_prob) { use_real_prob_ = use_real_prob; }
  void usePrior(bool use_prior) { use_prior_ = use_prior; }

  virtual void Serialize(SerializeHelper &helper) {
    XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Serialize(helper);
    prior_.Serialize(helper);
    helper.Serialize(use_real_prob_);
    helper.Serialize(use_prior_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Deserialize(helper);
    prior_.Deserialize(helper);
    helper.Deserialize(use_real_prob_);
    helper.Deserialize(use_prior_);
  }
};

template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
void xnnLogSoftmaxLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::readKaldiPu(std::istream &is) {
  using namespace xnnKaldiUtility;
  ReadVector(is, prior_);
  // convert to log
  prior_.Log();
}

template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
void xnnLogSoftmaxLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::readKaldiNnet1Pu(std::istream &is) {
  using namespace xnnKaldiUtility;
  int64 total_num = 0;
  std::vector<int32> vpdf;
  int32 pdf;
  while (is.peek() != ']') {
    is >> pdf;
    vpdf.push_back(pdf);
    total_num += pdf;
    is.get();
  }

  prior_.Resize(vpdf.size(), 1);
  for (size_t i = 0; i < vpdf.size(); i++) {
    prior_.Col(0)[i] = (float)vpdf[i] / total_num;
  }
  // convert to log
  prior_.Log();
}

}
#endif
