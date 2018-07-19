#ifndef FE_RONTEND_COMPONENT_DELTA_H_
#define FE_RONTEND_COMPONENT_DELTA_H_
#include "fe/frontend_component.h"

namespace idec {

class FrontendComponent_Delta : public FrontendComponentInterface {
 protected:
  int window_;
  int order_;
  std::vector<float> scales_;

 public:

  FrontendComponent_Delta(ParseOptions &po, int order,
                          const std::string name = "Delta") : FrontendComponentInterface(po, name),
    order_(order) {
    // default control parameters
    window_ = 2;

    // get parameters from config
    po.Register(name_ + "::delta-window", &window_,
                "Parameter controlling window for delta computation (actual window"
                " size for each delta order is 1 + 2*delta-window-size)");
  }

  virtual bool ReceiveOneFrameFromPrecedingComponent(FrontendComponentInterface
      *from, const float *data, size_t dim) {
    IDEC_ASSERT(input_map_[from] == 0);

    xnnFloatRuntimeMatrixCircularBuffer &input_buf = input_buf_[0];
    bool ret = true;
    if (input_buf.Empty()) {
      // first frame, must have window*order_ + 1 free space
      if ((int)input_buf.NumEmpty() < window_ * order_ + 1)
        return(false);

      // add the first frame for #window times
      for (int i = 0; i < window_*order_; ++i) {
        ret &= input_buf.PushbackOneColumn(data, dim);
      }
    }

    if (input_buf.NumEmpty() < 1)
      return(false);
    ret &= input_buf.PushbackOneColumn(data, dim);

    return(true);
  }

  virtual void Init() {
    FrontendComponentInterface::Init();

    // prepare output-related staff
    output_dim_ = input_dim_;
    output_buff_.Resize(output_dim_, 1);

    // check
    if (window_ < 1)
      IDEC_ERROR << "delta window size " << window_ <<
                 " must be greater than or equal to 1";

    std::vector<float> scales;
    scales.assign(1, 1.0f);

    for (int i = 1; i <= order_; i++) {
      std::vector<float> &prev_scales = scales,
                          &cur_scales = scales_;
      int window = window_;  // this code is designed to still
      // work if instead we later make it an array and do opts.window[i-1],
      // or something like that. "window" is a parameter specifying delta-window
      // width which is actually 2*window + 1.
      IDEC_ASSERT(window != 0);
      int prev_offset = (static_cast<int>(prev_scales.size() - 1)) / 2,
          cur_offset = prev_offset + window;
      cur_scales.assign(prev_scales.size() + 2 * window, 0);  // also zeros it.

      float normalizer = 0.0;
      for (int j = -window; j <= window; j++) {
        normalizer += j*j;
        for (int k = -prev_offset; k <= prev_offset; k++) {
          cur_scales[j + k + cur_offset] +=
            static_cast<float>(j)* prev_scales[k + prev_offset];
        }
      }

      for (int i = 0; i < (int)cur_scales.size(); ++i) {
        cur_scales[i] /= normalizer;
      }

      prev_scales = cur_scales;
    }
  }

  virtual bool Process() {
    if (input_buf_.empty())
      return(false);

    xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[0];
    while ((int)input_buff.NumCols() >= 2 * window_*order_ + 1) {
      output_buff_.SetZero();

      for (int t = 0; t < 2 * window_*order_ + 1; ++t) {
        if (scales_[t] == 0)
          continue; // no need to do anything

        for (int d = 0; d < output_dim_; ++d) {
          output_buff_.Col(0)[d] += scales_[t] * input_buff.Col(t)[d];
        }
      }

      // push one processed frame to succeeding components, return on error
      if (!SendOneFrameToSucceedingComponents())
        return(false);

      input_buff.PopfrontOneColumn();
    }

    return(true);
  }

  virtual bool Finalize() {
    IDEC_ASSERT(!input_buf_.empty());

    xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[0];

    bool ret = true;
    if (!input_buff.Empty()) {
      // push last frame for window_ times
      for (int i = 0; ret && i < window_*order_; ++i)
        ret &= input_buff.PushbackOneColumn(input_buff.Col(input_buff.NumCols() - 1),
                                            input_dim_);
    }
    return(ret & Process());
  }
};

}
#endif


