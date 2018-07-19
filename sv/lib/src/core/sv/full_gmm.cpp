#include "full_gmm.h"
#include "matrix_transform.h"

FullGmm::FullGmm(const UbmResource &fgmm):weights_(fgmm.weight),
  weights_xnn_(fgmm.weight_xnn), g_consts_(fgmm.g_const),
  g_consts_xnn_(fgmm.g_const_xnn), inv_covars_(fgmm.inv_covars),
  inv_covars_xnn_(fgmm.inv_covars_xnn), means_invcovars_(fgmm.means_invcovars),
  means_invcovars_xnn_(fgmm.means_invcovars_xnn), mean_invcovars_view_(fgmm.means_invcovars_xnn) {
  //this->weights_ = fgmm.weight;
  //this->weights_xnn_ = fgmm.weight_xnn;
  //this->g_consts_ = fgmm.g_const;
  //this->g_consts_xnn_ = fgmm.g_const_xnn;
  //this->inv_covars_ = fgmm.inv_covars;
  //this->inv_covars_xnn_ = fgmm.inv_covars_xnn;
  //this->means_invcovars_ = fgmm.means_invcovars;
  //this->means_invcovars_xnn_ = fgmm.means_invcovars_xnn;
  dim_ = means_invcovars_.Cols();
  mixture_ = weights_.Size();
  data_xnn_.Resize(dim_, 1);
  data_sq_.Resize(dim_, dim_);
  square_.Resize(dim_, dim_);
  
}

void FullGmm::Clear() {
  //this->weights_.Resize(0);
  //this->means_invcovars_.Resize(0, 0);
  //this->inv_covars_.clear();
}

void FullGmm::LogLikelihoodsPreselect(const DoubleVector &data,
                                      const vector<int> &indices, DoubleVector &loglikes) {
  idec::IDEC_ASSERT(Dim() == data.Size());
  //idec::xnnFloatRuntimeMatrix data_xnn, data_sq, square;
  MatrixTransform::EigenVector2XnnMatrix(data, data_xnn_);
  //MatrixTransform::EigenVector2XnnMatrix(loglikes, loglikes_xnn);
  //const Eigen::VectorXd *data_vecxd = data.Pointer();
  //Eigen::MatrixXd data_sq = (*data_vecxd) * data_vecxd->transpose();
  //data_sq.noalias() = data_sq * 0.5;
  //data_sq.Resize(data.Size(), data.Size());
  //square.Resize(data.Size(), data.Size());
  data_sq_.Covariance(data_xnn_,  0.5);

  int idx, num_indices;
  double trace, dot_product;
  //const Eigen::MatrixXd *inv_covars_idx = NULL, *means_invcovars_matxd = NULL;
  num_indices = static_cast<int>(indices.size());
  loglikes.Resize(num_indices);
  //idec::xnnRuntimeColumnMatrixView<idec::xnnFloatRuntimeMatrix>mean_invcovars_view(means_invcovars_xnn_);
  //means_invcovars_matxd = means_invcovars_.Pointer();
  for (int i = 0; i < num_indices; i++) {
    idx = indices[i];
    //dot_product = data_vecxd->dot(means_invcovars_matxd->row(idx));
    mean_invcovars_view_.ColView(idx, 1);
    dot_product = data_xnn_.DotProduct(mean_invcovars_view_);
    //inv_covars_idx = inv_covars_[idx].Pointer();
    //Eigen::MatrixXd square = data_sq * (*inv_covars_idx);
    square_.ScalePlusMatTMat(0.0, data_sq_, inv_covars_xnn_[idx]);
    trace = square_.Trace();
    //double sum_sq = square.diagonal().array().sum();
    //trace = (data_sq * (*inv_covars_idx)).trace();
    //cout << sum_sq << trace << endl;
    loglikes(i) = g_consts_(idx) + dot_product - trace;
  }
}

//void FullGmm::LogLikelihoodsPreselect(const DoubleVector &data,
//                                      const vector<int> &indices, DoubleVector &loglikes) const {
//  idec::IDEC_ASSERT(Dim() == data.Size());
//  const Eigen::VectorXd *data_vecxd = data.Pointer();
//  Eigen::MatrixXd data_sq = (*data_vecxd) * data_vecxd->transpose();
//  data_sq.noalias() = data_sq * 0.5;
//
//  int idx, num_indices;
//  double trace, dot_product;
//  const Eigen::MatrixXd *inv_covars_idx = NULL, *means_invcovars_matxd = NULL;
//  num_indices = static_cast<int>(indices.size());
//  loglikes.Resize(num_indices);
//  means_invcovars_matxd = means_invcovars_.Pointer();
//  for (int i = 0; i < num_indices; i++) {
//    idx = indices[i];
//	cout << means_invcovars_matxd->row(idx) << endl;
//    dot_product = data_vecxd->dot(means_invcovars_matxd->row(idx));
//	cout << dot_product << endl;
//    inv_covars_idx = inv_covars_[idx].Pointer();
//    Eigen::MatrixXd square = data_sq * (*inv_covars_idx);
//    //double sum_sq = square.diagonal().array().sum();
//    trace = (data_sq * (*inv_covars_idx)).trace();
//    cout << trace << endl;
//    loglikes(i) = g_consts_(idx) + dot_product - trace;
//	cout << loglikes(i) << endl;
//  }
//}

void FullGmm::GetMeans(DoubleMatrix &M) const {
  M.Resize(NumGauss(), Dim());
  DoubleMatrix covar;
  for (int i = 0; i < NumGauss(); i++) {
    covar = inv_covars_[i];
    covar.Invert();
    M.Row(i, covar * means_invcovars_.Rowv(i));
  }
}
