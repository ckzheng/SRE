#ifndef FE_RONTEND_COMPONENT_FILTERBANK2MFCC_H_
#define FE_RONTEND_COMPONENT_FILTERBANK2MFCC_H_
#include "fe/frontend_component.h"

namespace idec {


class FrontendComponent_Filterbank2Mfcc : public FrontendComponentInterface {
 protected:

  std::vector<float> lifter_coeffs_;
  xnnFloatRuntimeMatrix
  dct_matrix_;  // matrix we left-multiply by to perform DCT.

  xnnFloatRuntimeMatrix input_matrix_;

  int   num_ceps_;
  float cepstral_lifter_;

 public:

  bool use_energy_;

  FrontendComponent_Filterbank2Mfcc(ParseOptions &po,
                                    const std::string name = "Filterbank2Mfcc") : FrontendComponentInterface(po,
                                          name) {
    num_ceps_ = 13;
    cepstral_lifter_ = 22.0f;
    use_energy_ = false;

    // get parameters from config
    po.Register(name_ + "::num-ceps", &num_ceps_,
                "Number of cepstra in MFCC computation (including C0)");
    po.Register(name_ + "::cepstral-lifter", &cepstral_lifter_,
                "Constant that controls scaling of MFCCs");
  }


  void ComputeLifterCoeffs(float Q, std::vector<float> &coeffs);
  void ComputeDctMatrix(xnnFloatRuntimeMatrix *M);
  void MulMfccElements();

  virtual void Init() {
    FrontendComponentInterface::Init();

    // prepare output-related staff
    output_dim_ = num_ceps_;
    output_buff_.Resize(output_dim_, MAX_FRAME_RESERVED);


    // initial weight Matrix
    int num_bins = input_dim_;
    if (use_energy_) {
      num_bins = num_bins - 1;
    }

    // first generate large tmp matrix
    xnnFloatRuntimeMatrix tmp_matrix; // for calc dct_matrix_ use
    tmp_matrix.Resize(num_bins, num_bins);
    ComputeDctMatrix(&tmp_matrix);

    // copy part of matrix to dct_matrix_
    if (!use_energy_) {
      dct_matrix_ = tmp_matrix;
      dct_matrix_.Resize(input_dim_, num_ceps_);
    } else {
      dct_matrix_.Resize(input_dim_, num_ceps_);
      dct_matrix_.SetZero();

      for (int k = 1; k < num_ceps_; k++) {
#ifdef _MSC_VER
        memcpy_s(dct_matrix_.Col(k), num_bins * sizeof(float), tmp_matrix.Col(k),
                 num_bins * sizeof(float));
#else
        memcpy(dct_matrix_.Col(k), tmp_matrix.Col(k), num_bins * sizeof(float));
#endif
      }
      dct_matrix_.Col(0)[num_bins] = 1.0f;
    }


    // calc lifter_coffees
    if (cepstral_lifter_ != 0.0) {
      lifter_coeffs_.resize(num_ceps_);
      ComputeLifterCoeffs(cepstral_lifter_, lifter_coeffs_);
    }

  }




  virtual bool Process() {
    if (input_buf_.empty())
      return(false);

    // get input data from last component
    xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[0];

    if (input_buff.Empty())
      return(true);

    input_buff.GetNFrames(input_buff.NumCols(), input_matrix_);

    // do the DCT convert(filterbank 2 mfcc)
    output_buff_.Resize(output_dim_, input_matrix_.NumCols());
    output_buff_.ScalePlusMatTMat(0, dct_matrix_, input_matrix_);

    if (cepstral_lifter_ != 0.0)
      MulMfccElements();

    // TODO:  subtract_mean option,not support for now
    for (size_t t = 0; t < output_buff_.NumCols(); ++t) {
      if (!SendOneFrameToSucceedingComponents(output_buff_.Col(t)))
        return(false);
      input_buff.PopfrontOneColumn();
    }

    return(true);
  }

};


}
#endif


