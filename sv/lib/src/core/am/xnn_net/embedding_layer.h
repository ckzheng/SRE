#ifndef XNN_NET_EMBEDDING_LAYER_H_
#define XNN_NET_EMBEDDING_LAYER_H_
#include "am/xnn_net/layer_base.h"
namespace idec {


// Embedding layer [zhijie.yzj], map input ID to columns in W_
template<class InputMatrix, class OutputMatrix>
class xnnEmbeddingLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  OutputMatrix W_;

 public:
  xnnEmbeddingLayer() {};
  xnnEmbeddingLayer(const OutputMatrix &W) : W_(W) {};

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(W_.NumRows(), v.NumCols());
    for (size_t i = 0; i < v.NumCols(); i++) {
      size_t id = static_cast<size_t>(v.Col(i)[0] + 0.5f);
      if (id >= W_.NumCols()) {
        IDEC_ERROR << "embedding index out of range [" << id << " >= " << W_.NumCols();
      }

      // copy data
      for (size_t j = 0; j < u.NumRows(); j++) {
        u.Col(i)[j] = W_.Col(id)[j];
      }
    }
  }

  void Assign_Wm(std::vector<float> &arr, size_t rows) {
    size_t cols = arr.size() / rows;
    W_.Resize(cols, rows);
    for (size_t i = 0; i < rows; i++) {
      std::vector<float>::iterator it = arr.begin() + i * cols;
      std::copy(it, it + cols, stdext::make_unchecked_array_iterator(W_.Col(i)));
    }
  }

  virtual size_t vDim() const { return 1; }
  virtual size_t uDim() const { return W_.NumRows(); }

  //WMatrix* getW_() { return &W_; }
  //BMatrix* getb_() { return &b_; }

  virtual XnnLayerType getLayerType() const { return embeddingLayer; }
  virtual XnnMatrixType getMatrixType() const;

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    W_.Serialize(helper);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    W_.Deserialize(helper);
  }
};

};

#endif
