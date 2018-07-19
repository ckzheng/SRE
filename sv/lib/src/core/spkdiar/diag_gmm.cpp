#include "diag_gmm.h"
#include <fstream>
#include "new_matrix.h"
#include "new_vector.h"
#include "full_gmm.h"
#include "base/log_message.h"
#include <cmath>

using namespace std;

DiagGmm::DiagGmm(DoubleVector &weights, DoubleVector &gconsts,
                 DoubleMatrix &inv_vars,
                 DoubleMatrix &means_invvars) {
  this->weights_ = weights;
  this->gconsts_ = gconsts;
  this->inv_vars_ = inv_vars;
  this->means_invvars_ = means_invvars;
}

int  DiagGmm::Dim() const {
  return means_invvars_.Cols();
}

void DiagGmm::Resize(int nmix, int dim) {
  idec::IDEC_ASSERT(nmix > 0 && dim > 0);
  if (gconsts_.Size() != nmix) {
    gconsts_.Resize(nmix);
  }

  if (weights_.Size() != nmix) {
    weights_.Resize(nmix);
  }

  if (inv_vars_.Rows() != nmix || inv_vars_.Cols() != dim) {
    inv_vars_.Resize(nmix, dim);
    // must be initialized to unit for case of calling SetMeans while having
    // covars/invcovars that are not set yet (i.e. zero)
    inv_vars_.SetAllValues(1.0);
  }

  if (means_invvars_.Rows() != nmix || means_invvars_.Cols() != dim) {
    means_invvars_.Resize(nmix, dim);
  }
  update_gconst_ = false;
}

void DiagGmm::CopyFromFullGmm(const FullGmm &fullgmm) {
  int num_comp = fullgmm.NumGauss(), dim = fullgmm.Dim();
  Resize(num_comp, dim);
  memcpy(gconsts_.Data(), fullgmm.Gconsts().Data(), num_comp*sizeof(double));
  memcpy(weights_.Data(), fullgmm.Weights().Data(), num_comp*sizeof(double));
  DoubleMatrix means(num_comp, dim);
  fullgmm.GetMeans(means);
  int ncomp = NumGauss();
  DoubleMatrix covar;
  DoubleVector diag;
  const vector<DoubleMatrix> &inv_covar = fullgmm.InvCovars();
  for (int mix = 0; mix < ncomp; mix++) {
    covar = inv_covar[mix];
    covar.Invert();
    covar.DiagElements(diag);
    diag.InvertElements();
    inv_vars_.Row(mix, diag);
  }
  means_invvars_ = means;
  means_invvars_.MulElements(inv_vars_);
  ComputeGconsts();
}

int DiagGmm::ComputeGconsts() {
  int num_mix = NumGauss();
  int dim = Dim();
  const double M_LOG_2PI = 1.8378770664093454835606594728112;
  double offset = -0.5 * M_LOG_2PI * dim;  // constant term in gconst.
  int num_bad = 0;

  // Resize if Gaussians have been removed during Update()
  if (num_mix != static_cast<int>(gconsts_.Size())) {
    gconsts_.Resize(num_mix);
  }

  for (int mix = 0; mix < num_mix; mix++) {
    idec::IDEC_ASSERT(weights_(mix) >= 0);  // Cannot have negative weights.
    double gc = log(weights_(mix)) + offset;  // May be -inf if weights == 0
    for (int d = 0; d < dim; d++) {
      gc += 0.5 * log(inv_vars_(mix, d)) - 0.5 * means_invvars_(mix, d)
            * means_invvars_(mix, d) / inv_vars_(mix, d);
    }

    if (isnan(gc)) {  // negative infinity is OK but NaN is not acceptable
      idec::IDEC_ERROR << "At component " << mix
                       << ", not a number in gconst computation";
    }
    if (isinf(gc)) {
      num_bad++;
      // If positive infinity, make it negative infinity.
      // Want to make sure the answer becomes -inf in the end, not NaN.
      if (gc > 0) gc = -gc;
    }
    gconsts_(mix) = gc;
  }
  update_gconst_ = true;
  return num_bad;
}

// Outputs the per-component log-likelihoods.
void DiagGmm::LogLikelihoods(const DoubleVector &feat,
                             DoubleVector &loglikes)  const {
  int guass_num = gconsts_.Size();
  loglikes.Resize(guass_num);
  loglikes = gconsts_;
  if (feat.Size() != Dim()) {
    idec::IDEC_ERROR << "dimension not equal feat != dim";
  }
  DoubleVector feat_sq = feat;
  feat_sq.ApplyPow(2.0);
  loglikes += means_invvars_ * feat;
  loglikes += inv_vars_ * feat_sq * (-0.5);
}

void DiagGmm::LogLikelihoods(const DoubleMatrix &data,
                             DoubleMatrix &loglikes)  const {
  unsigned int frames_num = data.Rows();
  idec::IDEC_ASSERT(frames_num != 0);
  if ((frames_num != loglikes.Rows()) || (gconsts_.Size() != loglikes.Cols())) {
    loglikes.Resize(frames_num, gconsts_.Size());
  }
  for (int i = 0; i <frames_num; ++i) {
    loglikes.Row(i, gconsts_);
  }

  if (data.Cols() != Dim()) {
    idec::IDEC_ERROR << "DiagGmm::ComponentLogLikelihood, dimension "
                     << "mismatch " << data.Cols() << " vs. " << Dim();
  }

  DoubleMatrix data_sq(data);
  data_sq.ApplyPow(2.0);
  loglikes += data * means_invvars_.Transpose();
  loglikes += data_sq * inv_vars_.Transpose() * (-0.5);
}


double DiagGmm::LogLikelihood(const DoubleVector &data) const {
  if (!update_gconst_) {
    idec::IDEC_ERROR << "Must call ComputeGconsts() before computing likelihood";
  }
  DoubleVector loglikes;
  LogLikelihoods(data, loglikes);
  double log_sum = loglikes.LogSumExp();
  if (isnan(log_sum) || isinf(log_sum)) {
    idec::IDEC_ERROR << "Invalid answer (overflow or invalid variances/features?)";
  }
  return log_sum;
}

int DiagGmm::NumGauss() const {
  return weights_.Size();
}

/// Get gaussian selection information for one frame.
double DiagGmm::GaussianSelection(const DoubleVector &data, int num_gselect,
                                  std::vector<int> &output) const {
  int num_gauss = NumGauss();
  DoubleVector loglikes(num_gauss);
  output.clear();
  this->LogLikelihoods(data, loglikes);

  double thresh;
  if (num_gselect < num_gauss) {
    DoubleVector loglikes_copy(loglikes);
    double *ptr = loglikes_copy.Data();
    std::nth_element(ptr, ptr + num_gauss - num_gselect, ptr + num_gauss);
    thresh = ptr[num_gauss - num_gselect];
  } else {
    thresh = -std::numeric_limits<double>::infinity();
  }

  double tot_loglike = -std::numeric_limits<double>::infinity();
  std::vector<std::pair<double, int> > pairs;
  for (int p = 0; p < num_gauss; p++) {
    if (loglikes(p) >= thresh) {
      pairs.push_back(std::make_pair(loglikes(p), p));
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            std::greater<std::pair<double, int> >());
  for (int j = 0;
       j < num_gselect && j < static_cast<int>(pairs.size());
       j++) {
    output.push_back(pairs[j].second);
    tot_loglike = LogAdd(tot_loglike, pairs[j].first);
  }
  idec::IDEC_ASSERT(!output.empty());
  return tot_loglike;
}

double DiagGmm::GaussianSelection(const DoubleMatrix &data, int num_gselect,
                                  std::vector<std::vector<int> > &output) const {
  double ans = 0.0;
  int num_frames = data.Rows(), num_gauss = NumGauss();
  int max_mem = 10000000;
  // break up the utterance if needed.
  int mem_needed = num_frames * num_gauss * sizeof(double);
  if (mem_needed > max_mem) {
    // Break into parts and recurse, we don't want to consume too
    // much memory.
    int num_parts = (mem_needed + max_mem - 1) / max_mem;
    int part_frames = (data.Rows() + num_parts - 1) / num_parts;
    double tot_ans = 0.0;
    std::vector<std::vector<int> > part_output;
    output.clear();
    output.resize(num_frames);
    for (int p = 0; p < num_parts; p++) {
      int start_frame = p * part_frames,
          this_num_frames = std::min(num_frames - start_frame, part_frames);
      const DoubleMatrix &data_part = data.Crop(start_frame, 0, this_num_frames,
                                      data.Cols());
      tot_ans += GaussianSelection(data_part, num_gselect, part_output);
      for (int t = 0; t < this_num_frames; t++)
        output[start_frame + t].swap(part_output[t]);
    }
    idec::IDEC_ASSERT(!output.back().empty());
    return tot_ans;
  }

  idec::IDEC_ASSERT(num_frames != 0);
  DoubleMatrix loglikes_mat(num_frames, num_gauss);
  this->LogLikelihoods(data, loglikes_mat);

  output.clear();
  output.resize(num_frames);
  for (int i = 0; i < num_frames; i++) {
    const DoubleVector &loglikes = loglikes_mat.Rowv(i);
    double thresh;
    if (num_gselect < num_gauss) {
      DoubleVector loglikes_copy(loglikes);
      double *ptr = loglikes_copy.Data();
      std::nth_element(ptr, ptr + num_gauss - num_gselect, ptr + num_gauss);
      thresh = ptr[num_gauss - num_gselect];
    } else {
      thresh = -std::numeric_limits<double>::infinity();
    }

    double tot_loglike = -std::numeric_limits<double>::infinity();
    std::vector<std::pair<double, int> > pairs;
    for (int p = 0; p < num_gauss; p++) {
      if (loglikes(p) >= thresh) {
        pairs.push_back(std::make_pair(loglikes(p), p));
      }
    }
    std::sort(pairs.begin(), pairs.end(),
              std::greater<std::pair<double, int> >());
    std::vector<int> &this_output = (output)[i];
    for (int j = 0; j < num_gselect && j < static_cast<int>(pairs.size()); j++) {
      this_output.push_back(pairs[j].second);
      tot_loglike = LogAdd(tot_loglike, pairs[j].first);
    }
    idec::IDEC_ASSERT(!this_output.empty());
    ans += tot_loglike;
  }
  return ans;
}

void DiagGmm::Read(const string &mdl_path) {
  ResourceLoader res_loader;
  string label;
  ifstream ifs(mdl_path.c_str(), ios::binary);
  if (!ifs) {
    idec::IDEC_ERROR << "Open file " << mdl_path << " error!";
  }

  int size, cols, rows;
  if (ifs.peek() != '\0')
    idec::IDEC_ERROR << "only support kaldi binary format";
  ifs.get();

  if (ifs.peek() != 'B')
    idec::IDEC_ERROR << "only support kaldi binary format";
  ifs.get();

  label = "<DiagGMM>";
  res_loader.ReadLabel(ifs, label);
  label = "<GCONSTS>";
  res_loader.ReadLabel(ifs, label);
  res_loader.ReadLabel(ifs, "FV");
  res_loader.ReadInt(ifs, size);
  res_loader.ReadFVector(ifs, gconsts_, size);
  label = "<WEIGHTS>";
  res_loader.ReadLabel(ifs, label);
  res_loader.ReadLabel(ifs, "FV");
  res_loader.ReadInt(ifs, size);
  res_loader.ReadFVector(ifs, weights_, size);
  label = "<MEANS_INVVARS>";
  res_loader.ReadLabel(ifs, label);
  res_loader.ReadLabel(ifs, "FM");
  res_loader.ReadInt(ifs, rows);
  res_loader.ReadInt(ifs, cols);
  res_loader.ReadFMatrix(ifs, means_invvars_, rows, cols);
  label = "<INV_VARS>";
  res_loader.ReadLabel(ifs, label);
  res_loader.ReadLabel(ifs, "FM");
  res_loader.ReadInt(ifs, rows);
  res_loader.ReadInt(ifs, cols);
  res_loader.ReadFMatrix(ifs, inv_vars_, rows, cols);
  label = "</DiagGMM>";
  res_loader.ReadLabel(ifs, label);
}

void DiagGmm::GetMeans(DoubleMatrix &means) {
  DoubleMatrix vars = inv_vars_;
  means.Resize(inv_vars_.Rows(), inv_vars_.Cols());
  for (int i = 0; i < inv_vars_.Rows(); ++i) {
    for (int j = 0; j < inv_vars_.Cols(); ++j) {
      vars(i, j) = 1 / inv_vars_(i, j);
    }
  }

  for (int i = 0; i < inv_vars_.Rows(); ++i) {
    for (int j = 0; j < inv_vars_.Cols(); ++j) {
      means(i, j) = vars(i,j) * means_invvars_(i, j);
    }
  }
}

void DiagGmm::GetVars(DoubleMatrix &vars) {
  vars.Resize(inv_vars_.Rows(), inv_vars_.Cols());
  for (int i = 0; i < inv_vars_.Rows(); ++i) {
    for (int j = 0; j < inv_vars_.Cols(); ++j) {
      vars(i, j) = 1 / inv_vars_(i, j);
    }
  }
}

void DiagGmm::WriteNLSFormat(const string &mdl_path) {
  ofstream os(mdl_path.c_str(), std::ios::binary);
  DoubleMatrix mean;
  DoubleMatrix var;
  GetMeans(mean);
  GetVars(var);
  int num_mix = NumGauss();
  unsigned int dim = Dim();
  unsigned int num_mix_u = num_mix;


  int file_len = sizeof(num_mix)
                 + sizeof(num_mix_u)
                 + sizeof(float)*num_mix
                 + num_mix * (sizeof(dim) + sizeof(float)*dim + sizeof(dim) + sizeof(
                                float)*dim + sizeof(float));

  os.write((const char *)(&file_len), sizeof(file_len));
  os.write((const char *)(&num_mix), sizeof(num_mix));
  os.write((const char *)(&num_mix_u), sizeof(num_mix_u));

  // weight
  vector<float> wt_data;
  for (int i = 0; i < weights_.Size(); ++i) {
    wt_data.push_back(weights_(i));
  }

  os.write((const char *)(&wt_data[0]), sizeof(float)*num_mix);

  vector<float> mean_data, var_data;
  for (int m = 0; m < num_mix; m++) {
    for (int d = 0; d < dim; d++) {
      mean_data.push_back(mean(m, d));
      var_data.push_back(var(m, d));
    }

    float gconst = gconsts_(m);
    os.write((const char *)(&dim), sizeof(dim));
    os.write((const char *)(&mean_data[0]), sizeof(float)*dim);
    os.write((const char *)(&dim), sizeof(dim));
    os.write((const char *)(&var_data[0]), sizeof(float)*dim);
    os.write((const char *)(&gconst), sizeof(float));
	mean_data.clear();
	var_data.clear();
  }
  os.close();
  ComputeGconsts();
}

double DiagGmm::LogAdd(double  x, double  y) const {
  double diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  const double dbl_epsilon = 2.2204460492503131e-16;
  const double kMinLogDiffDouble = log(dbl_epsilon);
  // diff is negative.  x is now the larger one.
  if (diff >= kMinLogDiffDouble) {
    double res;
    res = x + log1pf(expf(x));
    return res;
  } else {
    // return the larger one.
    return x;
  }
}

