#ifndef XNN_NET_LAYER_BASE_H_
#define XNN_NET_LAYER_BASE_H_

#include <assert.h>
#include <string>
#include <fstream>

#include "base/serialize_helper.h"
#include "am/xnn_runtime.h"
#include "am/xnn_feature.h"
#include "am/xnn_kaldi_utility.h"

#ifndef _MSC_VER
namespace stdext {
template<class _Iterator> inline
_Iterator make_unchecked_array_iterator(
  _Iterator _Ptr) {   // construct with pointer
  return _Ptr;
}
}
#endif // MSC_VER

namespace idec {

const size_t kAmEvaluatorDefaultBlockSize = 8;
typedef xnnRuntimeColumnMatrixView<xnnFloatRuntimeMatrix> xnnFloatRuntimeMatrixView;
typedef xnnRuntimeColumnMatrixView<xnnFloat16RuntimeMatrix> xnnFloat16RuntimeMatrixView;
typedef xnnRuntimeColumnMatrixView<xnnFloat8RuntimeMatrix> xnnFloat8RuntimeMatrixView;

enum XnnLayerType {
  normalizationLayer, linearLayer,            sigmoidLayer,   reluLayer,
  logsoftmaxLayer,    pnormLayer,             normLayer,      embeddingLayer,
  blstmLayer,         projectedblstmLayer,    lstmLayer,      softmaxLayer,
  blocksoftmaxLayer,  convolutionalLayer,     maxpoolingLayer,rescaleLayer,
  addshiftLayer,      purereluLayer,          multiconvolution1dLayer,
};

enum XnnMatrixType { FloatFloat, ShortFloat, CharFloat };

template<class InputMatrix, class OutputMatrix>
class xnnLayerBase {
 protected:
  bool supportBlockEval_;

 public:
  xnnLayerBase() : supportBlockEval_(true) {};
  virtual ~xnnLayerBase() {};

  void enableBlockEval(bool isEnable) { supportBlockEval_ = isEnable; }
  bool isBlockEvalSupported() { return supportBlockEval_; }

  virtual void InitIntermediateStates(std::vector<void *> &intermediate_states) {
    return; // default is to do nothing
  };

  virtual void DeleteIntermediateStates(std::vector<void *> &intermediate_states) {
    return; // default is to do nothing
  };

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const = 0;
  virtual void forwardPropRange(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId = 0) const {
    IDEC_ERROR << "not implemented yet for this type of layer";
  }

  virtual size_t vDim() const = 0;            // dimension of input
  virtual size_t uDim() const = 0;            // dimension of output
  virtual XnnLayerType getLayerType() const = 0; // type of the layer
  virtual XnnMatrixType getMatrixType() const;   // type of the matrices (weight, bias, input, output, etc.)

  virtual void Serialize(SerializeHelper &helper) {
    helper.Serialize(supportBlockEval_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    helper.Deserialize(supportBlockEval_);
  }
};
};

#endif
