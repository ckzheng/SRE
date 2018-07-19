#include "ivector_estimator.h"
#include "base/time_utils.h"
#include "matrix_transform.h"

void IvectorEstimator::GetIvectorDistMean(const UtteranceStats &utt_stats,
    DoubleVector &linear, DoubleMatrix &quadratic) const {
  idec::xnnFloatRuntimeMatrix utt_gamma, utt_x, lin, quad;
  MatrixTransform::EigenVector2XnnMatrix(utt_stats.Gamma(), utt_gamma);
  MatrixTransform::EigenMatrix2XnnMatrix(utt_stats.X().Transpose(), utt_x);
  idec::xnnRuntimeColumnMatrixView<idec::xnnFloatRuntimeMatrix> utt_view(utt_x);
  lin.Resize(IvectorDim(), 1);
  quad.Resize(IvectorDim(), IvectorDim());
  
  int N = NumGauss();
  float *pGamma = utt_gamma.Col(0);
  float gamma[8];
  unsigned int i;
  for (i = 0; i + 8 < N; i += 8) {
    gamma[0] = pGamma[i];
    gamma[1] = pGamma[i + 1];
    gamma[2] = pGamma[i + 2];
    gamma[3] = pGamma[i + 3];
    gamma[4] = pGamma[i + 4];
    gamma[5] = pGamma[i + 5];
    gamma[6] = pGamma[i + 6];
    gamma[7] = pGamma[i + 7];

    if (gamma[0] != 0.0) {
      utt_view.ColView(i, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i], utt_view);
      quad.ScalePlusvElemProdv(gamma[0], iv_res_.U_xx[i]);
    }

    if (gamma[1] != 0.0) {
      utt_view.ColView(i + 1, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i + 1], utt_view);
      quad.ScalePlusvElemProdv(gamma[1], iv_res_.U_xx[i + 1]);
    }

    if (gamma[2] != 0.0) {
      utt_view.ColView(i + 2, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i + 2], utt_view);
      quad.ScalePlusvElemProdv(gamma[2], iv_res_.U_xx[i + 2]);
    }

    if (gamma[3] != 0.0) {
      utt_view.ColView(i + 3, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i + 3], utt_view);
      quad.ScalePlusvElemProdv(gamma[3], iv_res_.U_xx[i + 3]);
    }

    if (gamma[4] != 0.0) {
      utt_view.ColView(i + 4, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i + 4], utt_view);
      quad.ScalePlusvElemProdv(gamma[4], iv_res_.U_xx[i + 4]);
    }

    if (gamma[5] != 0.0) {
      utt_view.ColView(i + 5, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i + 5], utt_view);
      quad.ScalePlusvElemProdv(gamma[5], iv_res_.U_xx[i + 5]);
    }

    if (gamma[6] != 0.0) {
      utt_view.ColView(i + 6, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i + 6], utt_view);
      quad.ScalePlusvElemProdv(gamma[6], iv_res_.U_xx[i + 6]);
    }

    if (gamma[7] != 0.0) {
      utt_view.ColView(i + 7, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i + 7], utt_view);
      quad.ScalePlusvElemProdv(gamma[7], iv_res_.U_xx[i + 7]);
    }
  }

  for (; i < N; ++i) {
    gamma[0] = pGamma[i];
    if (gamma[0] != 0.0) {
      utt_view.ColView(i, 1);
      lin.PlusMatTMat(iv_res_.Sigma_inv_M_trans_xx[i], utt_view);
      quad.ScalePlusvElemProdv(gamma[0], iv_res_.U_xx[i]);      
    }
  }

  float *pLin = NULL, *pQuad = NULL;
  pLin = lin.Col(0);
  linear.Resize(IvectorDim());
  quadratic.Resize(IvectorDim(), IvectorDim());
  for (int i = 0; i < IvectorDim(); ++i) {
    linear(i) = pLin[i];
    pQuad = quad.Col(i);
    for (int j = 0; j < IvectorDim(); ++j) {
      quadratic(j, i) = pQuad[j];
    }
  }
}

void IvectorEstimator::GetIvectorDistPrior(const UtteranceStats &utt_stats,
    DoubleVector &linear,
    DoubleMatrix &quadratic) const {
  linear(0) += prior_offset_;
  DoubleVector v(quadratic.Cols());
  v.SetAllValues(1.0);
  quadratic.DiagElementsAdd(v);
}

void IvectorEstimator::GetIvectorDistribution(const UtteranceStats &utt_stats,
    DoubleVector &mean,
    DoubleMatrix *var) const {
  if (!IvectorDependentWeights()) {
    DoubleVector linear;
    DoubleMatrix quadratic;
    double time_start, time_end;
    //time_start = idec::TimeUtils::GetTimeMilliseconds();
    //GetIvectorDistMean_(utt_stats, linear, quadratic);
    //time_end = idec::TimeUtils::GetTimeMilliseconds();
    //cout << "before optimize " << time_end - time_start << " ms." << endl;
    time_start = idec::TimeUtils::GetTimeMilliseconds();
    GetIvectorDistMean(utt_stats, linear, quadratic);
    time_end = idec::TimeUtils::GetTimeMilliseconds();
    if (verbose_mode_) {
      idec::IDEC_INFO << "Calculate ivector mean spend " << time_end - time_start <<
                      " ms.";
    }

    GetIvectorDistPrior(utt_stats, linear, quadratic);
    if (var != NULL) {
      (*var) = quadratic;
      var->Invert();
      mean = (*var) * linear;
    } else {
      quadratic.Invert();
      mean = quadratic*linear;
    }
  }
}

void IvectorEstimator::Run(const UtteranceStats &utt_stats,
                           DoubleVector &ivector) const {
  if (ivector.Size() != IvectorDim()) {
    ivector.Resize(IvectorDim());
  }
  ivector(0) = PriorOffset();
  GetIvectorDistribution(utt_stats, ivector, NULL);
  ivector(0) -= PriorOffset();
}

bool IvectorEstimator::IvectorDependentWeights() const {
  return w_.Rows() != 0;
}
