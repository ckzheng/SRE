#include "am/xnn_runtime.h"
#include "am/xnn_kaldi_utility.h"
#include "base/log_message.h"
#include "util/parse-options.h"
#include "fe/frontend_component_filterbank2mfcc.h"

namespace idec {

/*functions define of Filterbank2Mfcc class */

void FrontendComponent_Filterbank2Mfcc::ComputeLifterCoeffs(float Q,
    std::vector<float> &coeffs) {
  // Compute liftering coefficients (scaling on cepstral coeffs)
  // coeffs are numbered slightly differently from HTK: the zeroth
  // index is C0, which is not affected.
  for (int i = 0; i < (int)coeffs.size(); i++) {
    coeffs[i] =(float) (1.0 + 0.5 * Q * sin (M_PI * i / Q));
  }

}


void FrontendComponent_Filterbank2Mfcc::ComputeDctMatrix(
  xnnFloatRuntimeMatrix *M) {
  //IDEC_ASSERT(M->NumRows() == M->NumCols());
  size_t K =   M->NumRows();
  size_t N = M->NumCols();

  IDEC_ASSERT(K > 0);
  IDEC_ASSERT(N > 0);
  float normalizer =(float) (std::sqrt(1.0 / static_cast<float>
                                       (N)));  // normalizer for
  // X_0.
  for (size_t j = 0; j < N; j++) {
    //(*M)(0, j) = normalizer;
    M->Col(0)[j] = normalizer;
  }

  normalizer = (float) (std::sqrt(2.0 / static_cast<int>
                                  (N)));  // normalizer for other
  // elements.
  for (size_t k = 1; k < K; k++) {
    for (size_t n = 0; n < N; n++) {
      // (*M)(k, n) = normalizer * std::cos( static_cast<double>(M_PI)/N * (n + 0.5) * k );
      M->Col(k)[n] = (float)(normalizer * std::cos( static_cast<double>(M_PI)/N *
                             (n + 0.5) * k ));
    }
  }
}



void FrontendComponent_Filterbank2Mfcc::MulMfccElements() {
  if (output_buff_.NumRows() != lifter_coeffs_.size())
    IDEC_ERROR << "lifter_coeffs length mismatch";

  for (size_t i = 0 ; i < output_buff_.NumCols(); i++) {
    for (size_t t= 0; t < lifter_coeffs_.size() ; t++) {
      output_buff_.Col(i)[t] *= lifter_coeffs_[t];

    }
  }
}
}