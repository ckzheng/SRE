#ifndef XNN_NET_MUTL_CONV_1D_LAYER_H_
#define XNN_NET_MUTL_CONV_1D_LAYER_H_

#include "am/xnn_net/linear_layer.h"

namespace idec {

// MultiCNN1d layer [yanping.wyp]
template<class WMatrix, class BMatrix, class InputMatrix, class OutputMatrix>
class xnnMultiConvolutional1DLayer : public xnnLayerBase
  < InputMatrix, OutputMatrix > {
 public:
  int32                              batch_num_;
  std::vector<std::vector<WMatrix> > W_;
  std::vector<BMatrix>               b_;

 public:
  xnnMultiConvolutional1DLayer() {};

  virtual void forwardProp(const InputMatrix &v, OutputMatrix &u,
                           std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), 1);
    InputMatrix v_padding;
    OutputMatrix cnn1d;
    OutputMatrix cnn1d_max;

    for (size_t i = 0, k = 0; i < W_.size(); ++i) {
      v_padding.Resize(v.NumRows(), v.NumCols() + 2 * W_[i][0].NumCols() - 2);
      v_padding.Padding(v, (int)W_[i][0].NumCols() - 1, (int)W_[i][0].NumCols() - 1);
      cnn1d.Resize(v_padding.NumCols() - W_[i][0].NumCols() + 1, 1);
      cnn1d_max.Resize(1, 1);

      for (size_t j = 0; j < W_[i].size(); ++j, ++k) {
        cnn1d.Convolution1d(v_padding, W_[i][j]);
        cnn1d_max.Max(cnn1d);
        u.Col(0)[k] = cnn1d_max.Col(0)[0] + b_[i].Col(0)[j];
      }
    }
    u.Tanh();
  }

  virtual XnnLayerType getLayerType() const { return multiconvolution1dLayer; }

  virtual size_t vDim() const {
    if (W_.size() > 0 && W_[0].size() > 0)
      return W_[0][0].NumRows();
    else
      return 0;
  }

  virtual size_t uDim() const {
    size_t udim = 0;
    for (size_t i = 0; i < W_.size(); ++i) {
      udim += W_[i].size();
    }
    return udim;
  }

  void readKaldiLayerNnet1(std::istream &is) {
    using namespace xnnKaldiUtility;
    bool binary = true;

    ExpectToken(is, binary, "<BatchNum>");
    ReadBasicType(is, binary, &batch_num_);
    W_.clear();
    b_.clear();
    for (size_t i = 0; i < batch_num_; ++i) {
      int32 kernel_size;
      int32 vec_len;
      std::vector<WMatrix> W_list;
      BMatrix b;

      ExpectToken(is, binary, "<KernelSize>");
      ReadBasicType(is, binary, &kernel_size);
      ExpectToken(is, binary, "<ConstSize>");
      ReadBasicType(is, binary, &vec_len);

      int peekval = is.peek();
      if (peekval != 'F')
        IDEC_ERROR << "Only uncompressed matrix supported";
      std::string token;
      ReadToken(is, binary, &token);
      if (token != "FM")
        IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;

      int32 kernel_num, cols;
      ReadBasicType(is, binary, &kernel_num);
      ReadBasicType(is, binary, &cols);

      for (size_t j = 0; j < kernel_num; ++j) {
        WMatrix W;
        W.Resize(vec_len, kernel_size);
        for (int32 k = 0; k < kernel_size; k++) {
          //is.read(reinterpret_cast<char*>(W.Col(k)), sizeof(Real)*vec_len);
          is.read(reinterpret_cast<char *>(W.Col(kernel_size - 1 - k)),
                  sizeof(Real)*vec_len);
          if (is.fail()) IDEC_ERROR << "read matrix error";
        }
        W_list.push_back(W);
      }

      peekval = is.peek();
      if (peekval != 'F')
        IDEC_ERROR << "Only uncompressed vector supported";
      ReadToken(is, binary, &token);
      if (token != "FV")
        IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

      int32 size;
      ReadBasicType(is, binary, &size);
      b.Resize(size, 1);
      if (size > 0)
        is.read(reinterpret_cast<char *>(b.Col(0)), sizeof(Real)*size);
      if (is.fail()) IDEC_ERROR << "read bias error";

      W_.push_back(W_list);
      b_.push_back(b);
    }
  }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    helper.Serialize(batch_num_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    helper.Deserialize(batch_num_);
  }
};

};

#endif
