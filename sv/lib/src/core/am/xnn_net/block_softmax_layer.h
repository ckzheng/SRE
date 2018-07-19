#ifndef _XNN_NET_BLOCK_SOFTMAX_LAYER_H
#define _XNN_NET_BLOCK_SOFTMAX_LAYER_H
#include "am/xnn_net/linear_layer.h"
#include "am/xnn_net/softmax_layer.h"

namespace idec {

// BlockSoftmax layer [mandy.mzy]
template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
class xnnBlockSoftmaxLayer : public XnnLinearLayer< WMatrix, BMatrix, InputMatrix, OutputMatrix > {
  friend class xnnBlockSoftmaxLayer< xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix >;
  friend class xnnBlockSoftmaxLayer< xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix >;

 protected:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::W_;
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::b_;

 public:
  using XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::uDim;

  xnnBlockSoftmaxLayer() : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>() {};
  xnnBlockSoftmaxLayer(const XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix> &layer) {
    *dynamic_cast<XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>*>(this) = layer; 
  };

  xnnBlockSoftmaxLayer(const WMatrix &W,const BMatrix &b) : XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>(W, b) {};
  xnnBlockSoftmaxLayer(const XnnSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer);

  virtual XnnLayerType getLayerType() const { return blocksoftmaxLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      forwardPropRange(v, u, block_offset[bl], block_dims[bl], 0);
    }
  }

  virtual void forwardPropRange(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const;

  virtual void Serialize(SerializeHelper &helper) {
    XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Serialize(helper);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    XnnLinearLayer<WMatrix, BMatrix, InputMatrix, OutputMatrix>::Deserialize(helper);
  }

  virtual void ReadData(std::istream &is, bool binary) {
    xnnKaldiUtility::ReadIntegerVector(is, binary, &block_dims);
    block_offset.resize(block_dims.size() + 1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i + 1] = block_offset[i] + block_dims[i];
    }

    // check
    if (uDim() != block_offset[block_offset.size() - 1]) {
      IDEC_ERROR << "sum of each block dim is not equal to output dim";
    }
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
};

};

#endif
