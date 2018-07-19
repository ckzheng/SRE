#ifndef XNN_NET_MAX_POOLING_LAYER_H_
#define XNN_NET_MAX_POOLING_LAYER_H_
#include "am/xnn_net/layer_base.h"

namespace idec {

// max-pooling layer [shaofei.xsf]
template<class InputMatrix, class OutputMatrix> class xnnMaxpoolingLayer :
  public xnnLayerBase < InputMatrix, OutputMatrix > {
 protected:
  size_t vdim_;
  size_t udim_;
  size_t pool_size_;
  size_t pool_step_;
  size_t pool_stride_;

 public:
  xnnMaxpoolingLayer() {};
  xnnMaxpoolingLayer(const size_t vdim, const size_t udim) : vdim_(vdim),
    udim_(udim) {
    pool_size_ = 0;
    pool_step_ = 0;
    pool_stride_ = 0;
  }

  xnnMaxpoolingLayer(const size_t vdim, const size_t udim,
                     const size_t pool_size, const size_t pool_step,
                     const size_t pool_stride) : vdim_(vdim), udim_(udim), pool_size_(pool_size),
    pool_step_(pool_step), pool_stride_(pool_stride) {};

  virtual XnnLayerType getLayerType() const { return maxpoolingLayer; }

  virtual void forwardProp(const InputMatrix &v /*input*/,
                           OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    size_t num_patches = vDim() / pool_stride_;
    size_t num_pools = 1 + (num_patches - pool_size_) / pool_step_;

    //u.Set(-1e20);
    for (size_t col = 0; col < v.NumCols(); col++) {
      for (size_t q = 0; q < num_pools; q++) {
        // get output buffer of the pool
        float *u_pool = u.Col(col);
        float *v_pool = v.Col(col);
        for (size_t str = 0; str < pool_stride_; str++) {
          float *out = u_pool + q*pool_stride_ + str;
          *out = (float)-1e20;
          for (size_t r = 0; r < pool_size_; r++) {
            size_t p = r + q * pool_step_;
            float *in = v_pool + p*pool_stride_ + str;
            *out = std::max(*in, *out);
          }
        }
      }
    }
  }

  virtual size_t vDim() const { return vdim_; }
  virtual size_t uDim() const { return udim_; }

  void setvDim(size_t vdim) { vdim_ = vdim; }
  void setuDim(size_t udim) { udim_ = udim; }

  void readKaldiLayerNnet1(std::istream &is/*, size_t nThread = 1*/) {
    using namespace xnnKaldiUtility;
    bool binary = true; // always use binary format

    int32 pool_size;
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<PoolSize>");
      ReadBasicType(is, binary, &pool_size);
    }
    pool_size_ = pool_size;

    int32 pool_step;
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<PoolStep>");
      ReadBasicType(is, binary, &pool_step);
    }
    pool_step_ = pool_step;

    int32 pool_stride;
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<PoolStride>");
      ReadBasicType(is, binary, &pool_stride);
    }
    pool_stride_ = pool_stride;
  }

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    helper.Serialize(vdim_);
    helper.Serialize(udim_);
    helper.Serialize(pool_size_);
    helper.Serialize(pool_step_);
    helper.Serialize(pool_stride_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    helper.Deserialize(vdim_);
    helper.Deserialize(udim_);
    helper.Deserialize(pool_size_);
    helper.Deserialize(pool_step_);
    helper.Deserialize(pool_stride_);
  }
};

};

#endif
