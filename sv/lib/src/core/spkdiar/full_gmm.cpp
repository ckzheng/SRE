#include "full_gmm.h"
#include "diag_gmm.h"

FullGmm::FullGmm(const UbmResource &fgmm) {
  this->weights_ = fgmm.weight;
  this->g_consts_ = fgmm.g_const;
  this->inv_covars_ = fgmm.inv_covars;
  this->means_invcovars_ = fgmm.means_invcovars;
}

void FullGmm::LogLikelihoodsPreselect(const DoubleVector &data,
                                      const vector<int> &indices, DoubleVector &loglikes) const {
  int dim = Dim();
  idec::IDEC_ASSERT(dim == data.Size());
  int num_indices = static_cast<int>(indices.size());
  loglikes.Resize(num_indices);
  // Initialize and make zero
  DoubleMatrix data_sq;
  const DoubleMatrix m_data(data);
  data_sq = m_data * m_data.Transpose();
  data_sq.Scale(0.5);

  double trace;
  DoubleMatrix dot_product;
  for (int i = 0; i < num_indices; i++) {
    int idx = indices[i];
    dot_product = means_invcovars_.Row(idx) * data;
    trace = (data_sq * inv_covars_[idx]).Trace();
    loglikes(i) = g_consts_(idx) + dot_product(0, 0) - trace;
  }
}

void FullGmm::GetMeans(DoubleMatrix &M) const {
  M.Resize(NumGauss(), Dim());
  DoubleMatrix covar;
  for (int i = 0; i < NumGauss(); i++) {
    covar = inv_covars_[i];
    covar.Invert();
    M.Row(i, covar * means_invcovars_.Rowv(i));
  }
}

int FullGmm::ComputeGconsts() {
  int num_mix = NumGauss(),
      dim = Dim();
  idec::IDEC_ASSERT(num_mix > 0 && dim > 0);
  const double M_LOG_2PI = 1.8378770664093454835606594728112;
  float offset = -0.5 * M_LOG_2PI * dim;  // constant term in gconst.
  int num_bad = 0;

  // Resize if Gaussians have been removed during Update()
  if (num_mix != g_consts_.Size()) {
    g_consts_.Resize(num_mix);
  }

  for (int mix = 0; mix < num_mix; mix++) {
    idec::IDEC_ASSERT(weights_(mix) >= 0);  // Cannot have negative weights.
    float gc = log(weights_(mix)) + offset;  // May be -inf if weights == 0
    DoubleMatrix covar(inv_covars_[mix]);
    covar.Invert();
    float logdet = covar.LogPosDefDet();
    DoubleVector val = covar * means_invcovars_.Rowv(mix);
    gc -= 0.5 * (logdet + means_invcovars_.Rowv(mix).Pointer()->dot(val()));

    if (std::isnan(gc)) {  // negative infinity is OK but NaN is not acceptable
      idec::IDEC_ERROR << "At component" << mix <<
                       ", not a number in gconst computation";
    }

    if (std::isinf(gc)) {
      num_bad++;
      if (gc > 0) gc = -gc;
    }
    g_consts_(mix) = gc;
  }

  update_gconst_ = true;
  return num_bad;
}

void FullGmm::CopyFromDiagGmm(const DiagGmm &diag_gmm) {
  g_consts_ = diag_gmm.Gconsts();
  weights_ = diag_gmm.Weights();
  means_invcovars_ = diag_gmm.MeanInvVars();
  int ncomp = NumGauss(), dim = Dim();
  for (int mix = 0; mix < ncomp; mix++) {
    inv_covars_[mix].Resize(dim, dim);
    for (int d = 0; d < dim; d++) {
      inv_covars_[mix](d, d) = diag_gmm.InvVars()(mix, d);
    }
  }
  ComputeGconsts();
}

void FullGmm::RemoveComponents(const vector<int> &gauss_in,
                               bool renorm_weights) {
  vector<int> gauss(gauss_in);
  std::unique(gauss.begin(), gauss.end());
  std::sort(gauss.begin(), gauss.end());

  // If efficiency is later an issue, will code this specially (unlikely,
  // except for quite large GMMs).
  for (size_t idx = 0; idx < gauss.size(); idx++) {
    idec::IDEC_ASSERT(idx < NumGauss());

    weights_.Erase(idx);
    g_consts_.Erase(idx);
    means_invcovars_.EraseRow(idx);
    inv_covars_.erase(inv_covars_.begin() + idx);
    if (renorm_weights) {
      float sum_weights = weights_.Sum();
      weights_.Scale(1.0 / sum_weights);
      update_gconst_ = false;
    }

    for (size_t j = idx + 1; j < gauss.size(); j++) {
      gauss[j]--;
    }
  }
}

void FullGmm::Split(int target_components, float perturb_factor,
                    vector<int> &history) {
  if (target_components <= NumGauss() || NumGauss() == 0) {
    idec::IDEC_WARN << "Cannot split from " << NumGauss() << " to "
                    << target_components << " components";
    return;
  }

  int current_components = NumGauss(), dim = Dim();
  //FullGmm *tmp = FullGmm();
  FullGmm tmp(*this);
  //tmp.CopyFromFullGmm(this);  // so we have copies of matrices...
  // First do the resize:
  weights_.Resize(target_components);
  const DoubleVector &tmp_weights = tmp.Weights();
  for (int i = 0; i < tmp_weights.Size(); ++i) {
    weights_(i) = tmp_weights(i);
  }
  //weights_.Range(0, current_components).CopyFromVec(tmp->weights_);
  means_invcovars_.Resize(target_components, dim);
  //means_invcovars_.Range(0, current_components, 0, dim).CopyFromMat(tmp->means_invcovars_);
  means_invcovars_.BlockAssign(0, 0, current_components, dim,
                               tmp.MeansInvCovars());
  //ResizeInvCovars(target_components, dim);
  inv_covars_.resize(target_components);
  const vector<DoubleMatrix> &tmp_inv_covars = tmp.InvCovars();
  for (int mix = 0; mix < current_components; mix++) {
    //inv_covars_[mix].CopyFromSp(tmp->inv_covars_[mix]);
    inv_covars_[mix] = tmp_inv_covars[mix];
  }

  //for (int mix = current_components; mix < target_components; mix++) {
  //  inv_covars_[mix].SetZero();
  //}

  g_consts_.Resize(target_components);

  //delete tmp;
  // future work(arnab): Use a priority queue instead?
  while (current_components < target_components) {
    float max_weight = weights_(0);
    int max_idx = 0;
    for (int i = 1; i < current_components; i++) {
      if (weights_(i) > max_weight) {
        max_weight = weights_(i);
        max_idx = i;
      }
    }

    // remember history
    history.push_back(max_idx);

    weights_(max_idx) *= 0.5;
    weights_(current_components) = weights_(max_idx);
    DoubleVector rand_vec(dim);
    for (int i = 0; i < dim; ++i) {
      rand_vec(i) = rand_generator_.Rand();
    }

    //rand_vec.Randn();
    //TpMatrix<float> invcovar_l(dim);
    //invcovar_l.Cholesky(inv_covars_[max_idx]);
    //rand_vec.MulTp(invcovar_l, kTrans);
    DoubleMatrix cholesky_matrix;
    inv_covars_[max_idx].CholeskyDecompose(cholesky_matrix);
    rand_vec = cholesky_matrix * rand_vec;

    //inv_covars_[current_components].CopyFromSp(inv_covars_[max_idx]);
    inv_covars_[current_components] = inv_covars_[max_idx];
    //means_invcovars_.Row(current_components).CopyFromVec(means_invcovars_.Row(max_idx));
    //means_invcovars_.Row(current_components).AddVec(perturb_factor, rand_vec);
    //means_invcovars_.Row(max_idx).AddVec(-perturb_factor, rand_vec);
    const DoubleVector &means_inv_covars_row1 = means_invcovars_.Rowv(
          max_idx) + rand_vec * perturb_factor;
    const DoubleVector &means_inv_covars_row2 = means_invcovars_.Rowv(
          max_idx) + rand_vec * perturb_factor;
    means_invcovars_.Row(current_components, means_inv_covars_row1);
    means_invcovars_.Row(max_idx, means_inv_covars_row2);
    current_components++;
  }
  ComputeGconsts();
}
