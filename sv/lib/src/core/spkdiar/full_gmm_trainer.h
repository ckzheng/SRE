#ifndef _FULL_GMM_TRAINER_H_
#define _FULL_GMM_TRAINER_H_

#include "diag_gmm.h"
#include "full_gmm_normal.h"

struct MleFullGmmOptions {
  double min_gaussian_weight;
  double min_gaussian_occupancy;
  double variance_floor;
  double max_condition;
  bool remove_low_count_gaussians;

  MleFullGmmOptions() {
    min_gaussian_weight = 1.0e-05;
    min_gaussian_occupancy = 100.0;
    variance_floor = 0.001;
    max_condition = 1.0e+04;
    remove_low_count_gaussians = true;
  }
};

class  FullGmmTrainer {
 public:
  FullGmmTrainer();
  ~ FullGmmTrainer();
  int Dim() const {return 0;}
  void AccumulateForComponent(const DoubleVector &data, int comp_index,
                              double weight) {
    idec::IDEC_ASSERT(data.Size() == Dim());
    double wt = weight;
    occupancy_(comp_index) += wt;
    const int kGmmMeans = 0x001, kGmmVariances = 0x002, kGmmWeights = 0x004;
    const Eigen::VectorXd *data_xd = data.Pointer();
    if (gmm_update_flags_ & kGmmMeans) {
      mean_accumulator_[comp_index] += data * wt;
      if (gmm_update_flags_ & kGmmVariances) {
        Eigen::MatrixXd *covar_acc_xd = covariance_accumulator_[comp_index].Pointer();
        (*covar_acc_xd).noalias() += (*data_xd) * data_xd->transpose() * wt;
      }
    }
  }

  void MleFullGmmUpdate(const MleFullGmmOptions &config,
                        int flags, FullGmm &gmm,
                        float *obj_change_out, float *count_out) {
    if (flags) {
      idec::IDEC_ERROR << "Flags in argument do not match the active accumulators";
    }

    gmm.ComputeGconsts();
    // float obj_old = MlObjective(*gmm, fullgmm_acc);

    int num_gauss = gmm.NumGauss();
    double occ_sum = occupancy_.Sum();

    int tot_floored = 0, gauss_floored = 0;

    // allocate the gmm in normal representation
    FullGmmNormal ngmm(gmm);

    std::vector<int> to_remove;
    DoubleVector weights(num_gauss);
    const vector<DoubleVector> &means = ngmm.Means();
    vector<DoubleVector> ngmm_means;
    vector<DoubleMatrix> ngmm_vars;
    for (int i = 0; i < num_gauss; i++) {
      double occ = occupancy_(i);
      double prob;
      if (occ_sum > 0.0)
        prob = occ / occ_sum;
      else
        prob = 1.0 / num_gauss;

      if (occ > static_cast<double> (config.min_gaussian_occupancy)
          && prob > static_cast<double> (config.min_gaussian_weight)) {

        weights(i) = prob;

        DoubleVector oldmean = means[i];
        const int kGmmMeans = 0x001, kGmmVariances = 0x002, kGmmWeights = 0x004;
        if (flags & (kGmmMeans | kGmmVariances)) {
          DoubleVector mean = mean_accumulator_[i];
          mean.Scale(1.0 / occ);
          ngmm_means[i] = mean;
        }

        if (flags & kGmmVariances) {
          idec::IDEC_ASSERT(flags & kGmmMeans);
          DoubleMatrix covar = covariance_accumulator_[i];
          covar.Scale(1.0 / occ);
          covar -= ngmm_means[i] * ngmm_means[i].Transpose();
          //covar.AddVec2(-1.0, ngmm.means_.Row(i));  // subtract squared means.
          // if we intend to only update the variances, we need to compensate by
          // adding the difference between the new and old mean
          if (!(flags & kGmmMeans)) {
            //oldmean.AddVec(-1.0, ngmm.means_.Row(i));
            oldmean -= ngmm_means[i];
            covar += oldmean * oldmean.Transpose();
            //covar.AddVec2(1.0, oldmean);
          }

          // Now flooring etc. of variance's eigenvalues.
          float floor = std::max(config.variance_floor,
                                 covar.MaxAbsEig() / config.max_condition);

          int floored = covar.EigenValueFloor(floor);

          if (floored) {
            tot_floored += floored;
            gauss_floored++;
          }

          // transfer to estimate
          ngmm_vars[i] = covar;
        }
      } else {  // Insufficient occupancy
        if (config.remove_low_count_gaussians &&
            static_cast<int>(to_remove.size()) < num_gauss - 1) {
          idec::IDEC_WARN << "Too little data - removing Gaussian (weight "
                          << std::fixed << prob
                          << ", occupation count " << std::fixed << occupancy_(i)
                          << ", vector size " << gmm.Dim() << ")";
          to_remove.push_back(i);
        } else {
          idec::IDEC_WARN << "Gaussian has too little data but not removing it because"
                          << (config.remove_low_count_gaussians ?
                              " it is the last Gaussian: i = "
                              : " remove-low-count-gaussians == false: i = ") << i
                          << ", occ = " << occupancy_(i) << ", weight = " << prob;
          weights(i) = std::max(prob, static_cast<double>(config.min_gaussian_weight));
        }
      }
    }

    // copy to natural representation according to flags
    ngmm.CopyToFullGmm(gmm, flags);
    gmm.ComputeGconsts();

    if (to_remove.size() > 0) {
      gmm.RemoveComponents(to_remove, true /* renorm weights */);
      gmm.ComputeGconsts();
    }

    if (tot_floored > 0)
      idec::IDEC_WARN << tot_floored << " variances floored in " << gauss_floored
                      << " Gaussians.";
  }

 private:
  int gmm_update_flags_;
  DoubleVector occupancy_;
  vector<DoubleVector > mean_accumulator_;
  vector<DoubleMatrix > covariance_accumulator_;
};

#endif // !_FULL_GMM_TRAINER_H_
