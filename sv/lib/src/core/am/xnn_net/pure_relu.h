#ifndef XNN_NET_PURE_RELU_LAYER_H_
#define XNN_NET_PURE_RELU_LAYER_H_
#include "am/xnn_net/linear_layer.h"
namespace idec {

// PureReLU layer [shaofei.xsf]
template<class InputMatrix, class OutputMatrix> class xnnPureReLULayer : public
  xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  size_t vdim_;
  size_t udim_;

 public:
  xnnPureReLULayer() {};
  xnnPureReLULayer(const size_t vdim, const size_t udim) : vdim_(vdim), udim_(udim) {};

  virtual XnnLayerType getLayerType() const { return purereluLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(v.NumRows(), v.NumCols());
    u = v;
    u.ReLU();
  }

  virtual size_t vDim() const { return vdim_; }
  virtual size_t uDim() const { return udim_; }

  void setvDim(size_t vdim) { vdim_ = vdim; }
  void setuDim(size_t udim) { udim_ = udim; }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    helper.Serialize(vdim_);
    helper.Serialize(udim_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    helper.Deserialize(vdim_);
    helper.Deserialize(udim_);
  }
};

};

#endif
