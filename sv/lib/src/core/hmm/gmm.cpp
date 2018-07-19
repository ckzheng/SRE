#include "gmm.h"
#include "base/idec_common.h"
#include "base/log_message.h"
#include "kaldi/io-funcs.h"
#include "kaldi/kaldi-vector.h"
#include  <cmath>
#include <algorithm>

#ifdef USE_SSE
#include "mmintrin.h"
#endif

#ifdef USE_NEON
#include <arm_neon.h>
inline float getsum(float32x4_t sum) {
  float32x2_t high = vget_high_f32(sum);
  float32x2_t low = vget_low_f32(sum);
  float32x2_t result = vpadd_f32(high, low);
  float ret = vget_lane_f32(result, 0) + vget_lane_f32(result, 1);
  return ret;
}
#endif


namespace alsid {
using namespace std;
#define  MINLARG 2.45e-308

void GaussianComponent::FixGConst() {
  float z;
  float sum = float(var_.size()*log(M_2PI));
  for (int i = 0; i < (int)var_.size(); i++) {
    z = float(var_[i] <= MINLARG ? LOG_ZERO : log(var_[i]));
    sum += z;
  }
  gconst_ = sum;
}

void GaussianComponent::ApplyVarFloor(const vector<float> &var_floor) {
  int num_floor = 0;

  for (int i = 0; i < (int)var_.size(); i++) {
    if (var_[i] < var_floor[i]) {
      var_[i] = var_floor[i];
      num_floor++;
    }
  }

  if (num_floor != 0) {
    idec::IDEC_INFO << "ApplyVarFloor:" << num_floor << " out of" << var_.size() <<
                    "var floored";
  }
}

float GaussianComponent::OutP(const float *x, const unsigned int &dim) {
#ifdef USE_SSE
  unsigned int i;
  const float *p1, *p2, *p3;
  __m128 m1, m2, s1;

  s1 = _mm_setzero_ps();
  for (i = 0, p1 = &mean_[0], p2 = &x[0], p3 = &var_[0]; i + 4 <= dim; i += 4) {
    m1 = _mm_loadu_ps(p1 + i);
    m2 = _mm_loadu_ps(p2 + i);
    m1 = _mm_sub_ps(m1, m2);
    m1 = _mm_mul_ps(m1, m1);
    m2 = _mm_loadu_ps(p3 + i);
    m1 = _mm_div_ps(m1, m2);
    s1 = _mm_add_ps(s1, m1);
  }
  for (; i < dim; i++) {
    m1 = _mm_load_ss(p1 + i);
    m2 = _mm_load_ss(p2 + i);
    m1 = _mm_sub_ss(m1, m2);
    m1 = _mm_mul_ss(m1, m1);
    m2 = _mm_load_ss(p3 + i);
    m1 = _mm_div_ss(m1, m2);
    s1 = _mm_add_ss(s1, m1);
  }

  // add together
  s1 = _mm_hadd_ps(s1, s1);
  s1 = _mm_hadd_ps(s1, s1);

  float fl;
  _mm_store_ss(&fl, s1);  // fix for Linux
  return -0.5f*(fl + gconst_);

#elif defined USE_NEON
  const float *px = x;
  const float *pmean = &mean_[0];
  const float *pvar = &var_[0];
  unsigned int d = 0;
  float sum = gconst_;
  float32x4_t sumv = vdupq_n_f32(0.0f);
  for (; d + 4 <= dim; d += 4, px += 4, pmean += 4, pvar += 4) {
    float32x4_t xv = vld1q_f32(px);
    float32x4_t meanv = vld1q_f32(pmean);
    float32x4_t varv = vld1q_f32(pvar);
    float32x4_t xmmv = vsubq_f32(xv, meanv);
    xmmv = vmulq_f32(xmmv, xmmv);
    float32x4_t recp_varv = vrecpeq_f32(varv);
    sumv = vmlaq_f32(sumv, xmmv, recp_varv);
  }

  sum += getsum(sumv);
  for (; d < dim; d++, px++, pmean++, pvar++) {
    float xmm = *px - *pmean;
    sum += xmm*xmm / *pvar;
  }

  return -0.5f* sum;

#else
  float sum, xmm;

  sum = gconst_;
  for(unsigned int i=0; i<dim; i++) {
    xmm = x[i]-mean_[i];
    sum += xmm*xmm/var_[i];
  }
  return -0.5f*sum;
#endif
}

GaussianTrainer::GaussianTrainer() {
  gamma_ = 0;
  num_frame_ = 0;
}

void  GaussianTrainer::Reset(GaussianComponent &gaussian) {
  gamma_ = 0;
  num_frame_ = 0;
  gamma_x_.assign(gaussian.Mean().size(), 0);
  gamma_xx_.assign(gaussian.Mean().size(), 0);
}

void GaussianTrainer::Accumulate(GaussianComponent &gaussian, const float *x,
                                 unsigned int dim, const double p) {

  std::vector<float> &mean = gaussian.Mean();
  std::vector<float> &var = gaussian.Var();
  gamma_ += p;
  num_frame_++;
  for (unsigned int i = 0; i < dim; i++) {
    gamma_x_[i] += p * (x[i]);
    gamma_xx_[i] += p * (x[i] * x[i]);
  }
}

void GaussianTrainer::MleUpdate(GaussianComponent &gaussian,
                                const MapUpdateOption &opt) {
  if (gamma_ > 0) {
    std::vector<float> &mean = gaussian.Mean();
    std::vector<float> &var = gaussian.Var();
    for (int i = 0; i < (int)mean.size(); i++) {
      if (opt.update_mean) {
        mean[i] = float(gamma_x_[i] / gamma_);
      }
      if (opt.update_var) {
        var[i] = float(gamma_xx_[i] / gamma_);
        var[i] -= mean[i] * mean[i];
        assert(var[i] > 0);
      }
    }
    gaussian.FixGConst();
  }
}

void  GaussianTrainer::MapUpdate(GaussianComponent &gaussian,
                                 const MapUpdateOption &opt) {
  if (gamma_ > 0) {
    float alpha_mean = gamma_ / (gamma_ + opt.tau_mean); // eqn.(14)
    float alpha_var = gamma_ / (gamma_ + opt.tau_var);
    std::vector<float> &mean = gaussian.Mean();
    std::vector<float> &var = gaussian.Var();
    for (int i = 0; i < (int)mean.size(); i++) {
      float old_mean = mean[i];
      if (opt.update_mean) {
        mean[i] = float(alpha_mean*(gamma_x_[i] / gamma_) + (1 - alpha_mean) *
                        old_mean);
      }
      if (opt.update_var) {
#if 0
        var[i] = float(alpha_var*(gamma_xx_[i] / gamma_) + (1 - alpha_var)*
                       (var[i] + old_mean * old_mean));
        var[i] -= mean[i] * mean[i];
#else
        // Computing the variance around the updated mean; this is:
        // E( (x - mu)^2 ) = E( x^2 - 2 x mu + mu^2 ) =
        // E(x^2) + mu^2 - 2 mu E(x).
        float new_var = (gamma_xx_[i] / gamma_) + mean[i] * mean[i] - 2.0f*mean[i] *
                        (gamma_x_[i] / gamma_);
        // now var is E(x^2) + m^2 - 2 mu E(x).
        // Next we do the appropriate weighting using the tau value.
        var[i] = alpha_var * new_var + (1 - alpha_var)*var[i];

#endif
        if (var[i] < -1.0) {
          idec::IDEC_WARNING << "must be wrong with the update equation";
        }
        var[i] = std::max((float)VAR_FLOOR, var[i]);
        //assert(var[i] > 0);
      }
    }
    gaussian.FixGConst();
  }
}

void  GaussianTrainer::MapUpdate2(GaussianComponent &gaussian, double weight,
                                  const MapUpdateOption &opt) {
  if (gamma_ > 0) {
    //float alpha_mean = gamma_ / (gamma_ + opt.tau_mean); // eqn.(14)
    //float alpha_var = gamma_ / (gamma_ + opt.tau_var);
    std::vector<float> &mean = gaussian.Mean();
    std::vector<float> &var = gaussian.Var();
    float old_mean, alpha_mean;
    for (int i = 0; i < mean.size(); i++) {
      old_mean = mean[i];
      if (opt.update_mean) {
        alpha_mean = opt.tau_mean;
        mean[i] = ((1 - alpha_mean) * gamma_x_[i] * gamma_ + alpha_mean * old_mean *
                   weight) / ((1 - alpha_mean) * gamma_ + alpha_mean * weight);
      }

      if (opt.update_var) {
        idec::IDEC_INFO << "No need to update variance.";
      }
      gaussian.FixGConst();
    }
  }
}

void  GaussianComponent::Serialize(idec::SerializeHelper &helper) {
  helper.Serialize(mean_);
  helper.Serialize(var_);
  helper.Serialize(gconst_);
}

void  GaussianComponent::Deserialize(idec::SerializeHelper &helper) {
  helper.Deserialize(mean_);
  helper.Deserialize(var_);
  helper.Deserialize(gconst_);
  FixGConst();
}

void GMM::OutP(const float *x, const unsigned int &dim,
               float *likelihood_per_mix, float *likelihood) {
  *likelihood = LOG_ZERO;
  for (unsigned int m = 0; m < mixture_.size(); m++) {
    if (wt_[m] > MINMIX) {
      likelihood_per_mix[m] = mixture_[m].OutP(x, dim) + wt_log_[m];
      *likelihood = idec::LogAdd(*likelihood, likelihood_per_mix[m]);
    } else {
      likelihood_per_mix[m] = LOG_ZERO;
    }
  }
}

vector<float> &GMM::getMean(unsigned int i) {
  return mixture_[i].Mean();
}

vector<float> &GMM::getVar(unsigned int i) {
  return mixture_[i].Var();
}

float GMM::OutP(const float *x, const unsigned int &dim) {
  float likelihood = LOG_ZERO;
  likelihood_.resize(mixture_.size());
  OutP(x, dim, &(likelihood_[0]), &likelihood);
  return likelihood;
}

void  GMM::Serialize(idec::SerializeHelper &helper) {
  int num_mix = static_cast<int>(mixture_.size());
  helper.Serialize(num_mix);

  helper.Serialize(wt_);
  for (unsigned int m = 0; m < mixture_.size(); m++) {
    mixture_[m].Serialize(helper);
  }
}

void  GMM::Deserialize(idec::SerializeHelper &helper) {
  int num_mix = 0;
  helper.Deserialize(num_mix);

  helper.Deserialize(wt_);
  mixture_.resize(num_mix);
  for (unsigned int m = 0; m < mixture_.size(); m++) {
    mixture_[m].Deserialize(helper);
  }

  wt_log_.resize(wt_.size());
  for (unsigned int m = 0; m < wt_.size(); m++) {
    wt_log_[m] = log(wt_[m]);
  }
}

IDEC_RETCODE GMM::Deserialize(const char *fname) {
  idec::SerializeHelper helper(100);
  IDEC_RETCODE ret = IDEC_SUCCESS;
  ret = helper.ReadFile(fname);
  if (IDEC_SUCCESS != ret) {
    return ret;
  }
  Deserialize(helper);
  return ret;
}

void GMM::FixGconst() {
  for (unsigned int m = 0; m < mixture_.size(); m++) {
    mixture_[m].FixGConst();
  }
}

GMMTrainer::GMMTrainer() {

}

void GMMTrainer::SetMdl(GMM *gmm) {
  gmm_ = gmm;
  gmm_trainer_.resize(gmm->NumMix());
  Reset();
}

void  GMMTrainer::Reset() {
  for (unsigned int m = 0; m < gmm_->NumMix(); m++) {
    gmm_trainer_[m].Reset(gmm_->mixture_[m]);
  }
}

void GMMTrainer::Accumulate(const float *x, unsigned int dim, const double p,
                            float *likelihood_per_mix, const float &likelihood, float acc_threshold) {
  double sum = 0;
  for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
    double posterior = exp(likelihood_per_mix[m] - likelihood);
    sum += posterior;
    if (posterior*p > acc_threshold) {
      gmm_trainer_[m].Accumulate(gmm_->mixture_[m], x, dim, posterior*p);
    }
  }

  assert(fabs(sum - 1) < 0.001);
}

void  GMMTrainer::MleUpdate(const MapUpdateOption &opt) {
  double gamma = 0;
  for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
    gamma += gmm_trainer_[m].gamma_;
  }

  // update mean & var
  for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
    if (gmm_trainer_[m].gamma_ / gamma > MINMIX) {
      gmm_trainer_[m].MleUpdate(gmm_->mixture_[m], opt);
    }
  }

  // update weight
  for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
    gmm_->wt_[m] = gmm_trainer_[m].gamma_ / gamma;
    gmm_->wt_[m] = std::max(gmm_->wt_[m], (float)MIX_FLOOR);
    gmm_->wt_log_[m] = log(gmm_->wt_[m]);
  }
}

void  GMMTrainer::MapUpdate2(const MapUpdateOption &opt) {
  float gamma = 0;
  for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
    gamma += gmm_trainer_[m].gamma_;
  }

  if (opt.update_mean || opt.update_var) {
    for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
      if (gmm_trainer_[m].gamma_ / gamma > MINMIX) {
        gmm_trainer_[m].MapUpdate2(gmm_->mixture_[m], gmm_->wt_[m], opt);
      }
    }
  }

  if (opt.update_weight) {
    idec::IDEC_INFO << "No need to update weight.";
  }
}

void  GMMTrainer::MapUpdate(const MapUpdateOption &opt) {
  float gamma = 0;
  for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
    gamma += gmm_trainer_[m].gamma_;
  }

  if (opt.update_mean || opt.update_var) {
    for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
      if (gmm_trainer_[m].gamma_ / gamma > MINMIX) {
        gmm_trainer_[m].MapUpdate(gmm_->mixture_[m], opt);
      }
    }
  }

  if (opt.update_weight) {
    float scale = 0;
    for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
      // problem if = gmm_trainer_[m].gamma_ ==0 &&  opt.tau_weight == 0
      float alpha_wt = 1.0;
      if (!idec::ApproximatelyEqnZero(opt.tau_weight)) {
        alpha_wt = gmm_trainer_[m].gamma_ / (gmm_trainer_[m].gamma_ + opt.tau_weight);
      }
      gmm_->wt_[m] = (gmm_trainer_[m].gamma_ *alpha_wt) / gamma +
                     (1 - alpha_wt)* gmm_->wt_[m];
      scale += gmm_->wt_[m];
    }

    // re-normalize weight sum to 1
    for (unsigned int m = 0; m < gmm_trainer_.size(); m++) {
      gmm_->wt_[m] /= scale;
      gmm_->wt_[m] = std::max(gmm_->wt_[m], (float)MIX_FLOOR);
      gmm_->wt_log_[m] = log(gmm_->wt_[m]);
    }
  }
}

HMM::HMM() {
  viterbi_prune_beam_ = -1;
}

void  HMM::InitFromUbm(const GMM &ubm, const unsigned int &num_state) {
  states_.reserve(num_state);
  states_.resize(0);
  for (unsigned int s = 0; s < num_state; s++) {
    states_.push_back(ubm);
  }
}

const vector<GMM> &HMM::GetStates() const {
  return states_;
}

void HMM::BeginEvaluate(float beam) {
  unsigned int S = NumState();
  viterbi_score_prev_.resize(S); // T*S [t,s] ==>t*S + s
  viterbi_score_.assign(S, LOG_ZERO); // T*S [t,s] ==>t*S + s
  decode_cur_frm_ = 0;
  viterbi_prune_beam_ = beam;
}

void HMM::Evaluate(const idec::xnnFloatRuntimeMatrix &feat, float *tol_lr,
                   std::vector< std::vector<int> > &trace_,
                   std::vector< std::vector<float> > &loglikelihood_) {
  unsigned int S = NumState();
  unsigned int M = states_[0].mixture_.size();
  unsigned int T = feat.NumCols();
  unsigned int dim = feat.NumRows();
  float prob;

  std::vector<int> trace;
  std::vector<float> loglikelihood;
  for (unsigned int t = 0; t < T; t++) {
    trace.assign(S, -1);
    loglikelihood.assign(S, LOG_ZERO);
    prob = LOG_ZERO;
    if (decode_cur_frm_ == 0) {
      unsigned int s = 0;
      prob = states_[s].OutP(feat.Col(t), dim);
      viterbi_score_[s] = prob;
      loglikelihood[s] = prob;
      trace[s] = 0;
    } else {
      viterbi_score_prev_.swap(viterbi_score_);
      viterbi_score_.assign(S, LOG_ZERO);

      float max_score_cur_frame = LOG_ZERO;
      for (unsigned int s = 0; s < S; s++) {
        prob = LOG_ZERO;
        if (viterbi_score_prev_[s] != LOG_ZERO ||
            (s>0 && viterbi_score_prev_[s - 1] != LOG_ZERO)) {
          if (s == 0) {
            viterbi_score_[s] = viterbi_score_prev_[s];
            trace[s] = 0;
          } else {
            viterbi_score_[s] = std::max(viterbi_score_prev_[s],
                                         viterbi_score_prev_[s - 1]);
            if (viterbi_score_prev_[s] > viterbi_score_prev_[s - 1]) {
              trace[s] = s;
            } else {
              trace[s] = s - 1;
            }
          }

          // add pruning
          prob = states_[s].OutP(feat.Col(t), dim);
          if (viterbi_prune_beam_ < 0
              || viterbi_score_[s] > max_score_cur_frame - viterbi_prune_beam_) {
            viterbi_score_[s] += prob;
            max_score_cur_frame = std::max(max_score_cur_frame, viterbi_score_[s]);
          } else {
            viterbi_score_[s] = LOG_ZERO;
            trace[s] = -1;
          }
          loglikelihood[s] = prob;
        }
      }

      // offline pruning
      if (viterbi_prune_beam_ > 0) {
        for (unsigned int s = 0; s < S; s++) {
          if (viterbi_score_[s] < max_score_cur_frame - viterbi_prune_beam_) {
            viterbi_score_[s] = LOG_ZERO;
            trace[s] = -1;
          }
        }
      }
    }

    trace_.push_back(trace);
    loglikelihood_.push_back(loglikelihood);
    decode_cur_frm_++;
  }
  *tol_lr = viterbi_score_[S - 1];
}

void HMM::Evaluate(const idec::xnnFloatRuntimeMatrix &feat, float *tol_lr) {
  unsigned int S = NumState();
  unsigned int M = states_[0].mixture_.size();
  unsigned int T = feat.NumCols();
  unsigned int dim = feat.NumRows();

  for (unsigned int t = 0; t < T; t++) {
    if (decode_cur_frm_ == 0) {
      unsigned int s = 0;
      viterbi_score_[s] = states_[s].OutP(feat.Col(t), dim);
    } else {
      viterbi_score_prev_.swap(viterbi_score_);
      viterbi_score_.assign(S, LOG_ZERO);

      float max_score_cur_frame = LOG_ZERO;
      for (unsigned int s = 0; s < S; s++) {
        if (viterbi_score_prev_[s] != LOG_ZERO ||
            (s>0 && viterbi_score_prev_[s - 1] != LOG_ZERO)) {
          if (s == 0) {
            viterbi_score_[s] = viterbi_score_prev_[s];
          } else {
            viterbi_score_[s] = std::max(viterbi_score_prev_[s],
                                         viterbi_score_prev_[s - 1]);
          }

          // add pruning
          if (viterbi_prune_beam_ < 0
              || viterbi_score_[s] > max_score_cur_frame - viterbi_prune_beam_) {
            viterbi_score_[s] += states_[s].OutP(feat.Col(t), dim);
            max_score_cur_frame = std::max(max_score_cur_frame, viterbi_score_[s]);
          } else {
            viterbi_score_[s] = LOG_ZERO;
          }
        }
      }

      // offline pruning
      if (viterbi_prune_beam_ > 0) {
        for (unsigned int s = 0; s < S; s++) {
          if (viterbi_score_[s] < max_score_cur_frame - viterbi_prune_beam_)
            viterbi_score_[s] = LOG_ZERO;
        }
      }
    }

    decode_cur_frm_++;
  }
  *tol_lr = viterbi_score_[S - 1];
}

// add by zhuozhu
void HMM::Evaluate(float **feat, int dims, int cols, float *tol_lr,
                   unsigned int start, unsigned int end) {
  unsigned int S = NumState();
  unsigned int M = states_[0].mixture_.size();
  unsigned int T = end - start;
  unsigned int dim = dims;

  for (unsigned int t = 0; t < T; t++) {
    if (decode_cur_frm_ == 0) {
      unsigned int s = 0;
      viterbi_score_[s] = states_[s].OutP(feat[t + start], dim);
    } else {
      viterbi_score_prev_.swap(viterbi_score_);
      viterbi_score_.assign(S, LOG_ZERO);

      float max_score_cur_frame = LOG_ZERO;
      for (unsigned int s = 0; s < S; s++) {
        if (viterbi_score_prev_[s] != LOG_ZERO ||
            (s>0 && viterbi_score_prev_[s - 1] != LOG_ZERO)) {
          if (s == 0) {
            viterbi_score_[s] = viterbi_score_prev_[s];
          } else {
            viterbi_score_[s] = std::max(viterbi_score_prev_[s],
                                         viterbi_score_prev_[s - 1]);
          }

          // add pruning
          if (viterbi_prune_beam_ < 0
              || viterbi_score_[s] > max_score_cur_frame - viterbi_prune_beam_) {
            viterbi_score_[s] += states_[s].OutP(feat[t + start], dim);
            max_score_cur_frame = std::max(max_score_cur_frame, viterbi_score_[s]);
          } else {
            viterbi_score_[s] = LOG_ZERO;
          }
        }
      }

      // offline pruning
      if (viterbi_prune_beam_ > 0) {
        for (unsigned int s = 0; s < S; s++) {
          if (viterbi_score_[s] < max_score_cur_frame - viterbi_prune_beam_)
            viterbi_score_[s] = LOG_ZERO;
        }
      }
    }

    decode_cur_frm_++;
  }
  *tol_lr = viterbi_score_[S - 1];
}

void HMM::Evaluate(const idec::xnnFloatRuntimeMatrix &feat, float *tol_lr,
                   unsigned int start, unsigned int end) {
  unsigned int S = NumState();
  unsigned int M = states_[0].mixture_.size();
  unsigned int T = end - start;
  unsigned int dim = feat.NumRows();

  for (unsigned int t = 0; t < T; t++) {
    if (decode_cur_frm_ == 0) {
      unsigned int s = 0;
      viterbi_score_[s] = states_[s].OutP(feat.Col(t+start), dim);
    } else {
      viterbi_score_prev_.swap(viterbi_score_);
      viterbi_score_.assign(S, LOG_ZERO);

      float max_score_cur_frame = LOG_ZERO;
      for (unsigned int s = 0; s < S; s++) {
        if (viterbi_score_prev_[s] != LOG_ZERO ||
            (s>0 && viterbi_score_prev_[s - 1] != LOG_ZERO)) {
          if (s == 0) {
            viterbi_score_[s] = viterbi_score_prev_[s];
          } else {
            viterbi_score_[s] = std::max(viterbi_score_prev_[s],
                                         viterbi_score_prev_[s - 1]);
          }

          // add pruning
          if (viterbi_prune_beam_ < 0
              || viterbi_score_[s] > max_score_cur_frame - viterbi_prune_beam_) {
            viterbi_score_[s] += states_[s].OutP(feat.Col(t + start), dim);
            max_score_cur_frame = std::max(max_score_cur_frame, viterbi_score_[s]);
          } else {
            viterbi_score_[s] = LOG_ZERO;
          }
        }
      }

      // offline pruning
      if (viterbi_prune_beam_ > 0) {
        for (unsigned int s = 0; s < S; s++) {
          if (viterbi_score_[s] < max_score_cur_frame - viterbi_prune_beam_)
            viterbi_score_[s] = LOG_ZERO;
        }
      }
    }

    decode_cur_frm_++;
  }
  *tol_lr = viterbi_score_[S - 1];
}

void HMM::Serialize(const char *fname) {
  idec::SerializeHelper helper(100);
  Serialize(helper);
  helper.WriteFile(fname);
}

void HMM::Serialize(idec::SerializeHelper &helper) {
  int num_state = static_cast<int>(states_.size());
  helper.Serialize(num_state);
  for (unsigned int s = 0; s < states_.size(); s++) {
    states_[s].Serialize(helper);
  }
}

IDEC_RETCODE HMM::Deserialize(const char *fname) {
  IDEC_RETCODE ret = IDEC_SUCCESS;
  idec::SerializeHelper helper(100);
  ret = helper.ReadFile(fname);
  if (ret != IDEC_SUCCESS) {
    return ret;
  }
  Deserialize(helper);

  return ret;
}

void  HMM::Deserialize(idec::SerializeHelper &helper) {
  int num_state = 0;
  helper.Deserialize(num_state);
  states_.resize(num_state);
  for (unsigned int s = 0; s < states_.size(); s++) {
    states_[s].Deserialize(helper);
  }
}

HMMTrainer::HMMTrainer(HMM &hmm) :hmm_(hmm) {
  states_.resize(hmm_.states_.size());
  for (unsigned int s = 0; s < hmm_.states_.size(); s++) {
    states_[s].SetMdl(&(hmm_.states_[s]));
  }
}

void HMMTrainer::Reset() {
  for (unsigned int s = 0; s < states_.size(); s++) {
    states_[s].Reset();
  }
}

void  HMMTrainer::Accumulate(const idec::xnnFloatRuntimeMatrix &feat,
                             float alpha_beam, float acc_threshold) {
  Forward(feat, alpha_beam);
  Backward(feat, alpha_beam, acc_threshold);
}

void  HMMTrainer::Accumulate(const idec::xnnFloatRuntimeMatrix &feat,
                             float alpha_beam, float acc_threshold, unsigned int start, unsigned int end) {
  Forward(feat, alpha_beam, start, end);
  Backward(feat, alpha_beam, acc_threshold, start, end);
}

void  HMMTrainer::Accumulate(float **feat, int dims, int cols,
                             float alpha_beam, float acc_threshold, unsigned int start, unsigned int end) {
  Forward(feat, dims, cols, alpha_beam, start, end);
  Backward(feat, dims, cols, alpha_beam, acc_threshold, start, end);
}

void  HMMTrainer::AccumulateEqualLength(const idec::xnnFloatRuntimeMatrix
                                        &feat, float acc_threshold) {
  unsigned int S = hmm_.NumState();
  unsigned int M = hmm_.states_[0].mixture_.size();
  unsigned int T = feat.NumCols();
  unsigned int dim = feat.NumRows();
  alpha_.assign(T*S, LOG_ZERO);
  gaussian_lr_.assign(T*S*M,
                      LOG_ZERO);  //[T][S][M], [t, s, m] == >t*S*M + s*M + m, maybe out-of-memory?
  gmm_lr_.assign(T*S, LOG_ZERO);

  for (unsigned int t = 0; t < T; t++) {
    unsigned int s = t / (static_cast<float>(T) / S);
    hmm_.states_[s].OutP(feat.Col(t), dim, &(gaussian_lr_[0]) + (t*S*M + s*M),
                         &(gmm_lr_[0]) + t*S + s);
    states_[s].Accumulate(feat.Col(t), dim, 1.0,
                          (&(gaussian_lr_[0]) + (t*S*M + s*M)), gmm_lr_[t*S + s], acc_threshold);
  }
}

void  HMMTrainer::MapUpdate(const MapUpdateOption &opt) {
  for (unsigned int s = 0; s < hmm_.states_.size(); s++) {
    states_[s].MapUpdate(opt);
  }
}

void  HMMTrainer::MleUpdate(const MapUpdateOption &opt) {
  for (unsigned int s = 0; s < hmm_.states_.size(); s++) {
    states_[s].MleUpdate(opt);
  }
}

void  HMMTrainer::Forward(const idec::xnnFloatRuntimeMatrix &feat,
                          float alpha_beam) {
  unsigned int S = hmm_.NumState();
  unsigned int M = hmm_.states_[0].mixture_.size();
  unsigned int T = feat.NumCols();
  unsigned int dim = feat.NumRows();
  alpha_.assign(T*S, LOG_ZERO);
  gaussian_lr_.assign(T*S*M,
                      LOG_ZERO);  //[T][S][M], [t, s, m] == >t*S*M + s*M + m, maybe out-of-memory?
  gmm_lr_.assign(T*S, LOG_ZERO);

  //[t, s] == >t*S + s
  for (unsigned int t = 0; t < T; t++) {
    double max_alpha = LOG_ZERO;
    if (t == 0) {
      unsigned int s = 0;
      hmm_.states_[s].OutP(feat.Col(t), dim, &(gaussian_lr_[0]) + (t*S*M + s*M),
                           &(gmm_lr_[0]) + t*S + s);
      alpha_[t*S + s] = gmm_lr_[t*S + s];
    } else {
      for (unsigned int s = 0; s < S; s++) {
        if (alpha_[(t - 1)*S + s] != LOG_ZERO ||
            (s>0 && alpha_[(t - 1)*S + s - 1] != LOG_ZERO)) {
          if (s == 0) {
            alpha_[t*S + s] = alpha_[(t - 1)*S + s];
          } else {
            alpha_[t*S + s] = idec::LogAdd(alpha_[(t - 1)*S + s],
                                           alpha_[(t - 1)*S + s - 1]);
          }

          // pruning
          if (alpha_beam <0 || alpha_[t*S + s] > max_alpha - alpha_beam) {
            hmm_.states_[s].OutP(feat.Col(t), dim, &(gaussian_lr_[0]) + (t*S*M + s*M),
                                 &(gmm_lr_[0]) + t*S + s);
            alpha_[t*S + s] += gmm_lr_[t*S + s];
            max_alpha = std::max(alpha_[t*S + s], max_alpha);
          } else {
            alpha_[t*S + s] = LOG_ZERO;
          }
        }
      }


      // pruning
      if (alpha_beam > 0) {
        for (unsigned int s = 0; s < S; s++) {
          if (alpha_[t*S + s] < max_alpha - alpha_beam) {
            alpha_[t*S + s] = LOG_ZERO;
          }
        }
      }
    } // end T
  }
}

void  HMMTrainer::Backward(const idec::xnnFloatRuntimeMatrix &feat,
                           float alpha_beam, float acc_threshold) {
  unsigned int S = hmm_.NumState();
  unsigned int M = hmm_.states_[0].mixture_.size();
  unsigned int T = feat.NumCols();
  unsigned int dim = feat.NumRows();
  beta_.assign(S, LOG_ZERO);


  for (int t = T - 1; t >= 0; t--) {
    if (t == T - 1) {
      unsigned int s = S - 1;
      assert(gmm_lr_[t*S + s] != LOG_ZERO);
      beta_[s] = gmm_lr_[t*S + s];
      double gamma = beta_[s] + alpha_[t*S + s] - gmm_lr_[t*S + s] - alpha_[(T - 1)*S
                     + S - 1];
      assert(fabs(gamma) < 0.001);
      states_[s].Accumulate(feat.Col(t), dim, exp(gamma),
                            (&(gaussian_lr_[0]) + (t*S*M + s*M)), gmm_lr_[t*S + s], acc_threshold);
    } else {
      beta_prev_ = beta_;
      beta_.assign(S, LOG_ZERO);
      double sum = LOG_ZERO;
      for (int s = S - 1; s >= 0; s--) {
        if (alpha_[t*S + s] != LOG_ZERO) {
          if (beta_prev_[s] != LOG_ZERO ||
              (s != S - 1 && beta_prev_[s + 1] != LOG_ZERO)) {
            if (s == S - 1) {
              beta_[s] = beta_prev_[s];
            } else {
              beta_[s] = idec::LogAdd(beta_prev_[s], beta_prev_[s + 1]);
            }

            if (gmm_lr_[t*S + s] == LOG_ZERO) {
              hmm_.states_[s].OutP(feat.Col(t), dim, &(gaussian_lr_[0]) + (t*S*M + s*M),
                                   &(gmm_lr_[0]) + t*S + s);
            }
            beta_[s] += gmm_lr_[t*S + s];

            double gamma = beta_[s] + alpha_[t*S + s] - gmm_lr_[t*S + s] - alpha_[(T - 1)*S
                           + S - 1];
            sum = idec::LogAdd(sum, gamma);
            if (gamma > LOGSMALL_F) {
              states_[s].Accumulate(feat.Col(t), dim, exp(gamma),
                                    (&(gaussian_lr_[0]) + (t*S*M + s*M)), gmm_lr_[t*S + s], acc_threshold);
            }
          }
        }
      }

      if (alpha_beam < 0) {
        assert(fabs(sum) < 0.1);
      }
    }
  }
}

void  HMMTrainer::Forward(float **feat, int dims, int cols,
                          float alpha_beam, unsigned int start, unsigned int end) {
  unsigned int S = hmm_.NumState();
  unsigned int M = hmm_.states_[0].mixture_.size();
  unsigned int T = (end - start + 1);
  unsigned int dim = dims;
  alpha_.assign(T*S, LOG_ZERO);
  gaussian_lr_.assign(T*S*M,
                      LOG_ZERO);  //[T][S][M], [t, s, m] == >t*S*M + s*M + m, maybe out-of-memory?
  gmm_lr_.assign(T*S, LOG_ZERO);

  //[t, s] == >t*S + s
  for (unsigned int t = 0; t < T; t++) {
    double max_alpha = LOG_ZERO;
    if (t == 0) {
      unsigned int s = 0;
      hmm_.states_[s].OutP(feat[t + start], dim, &(gaussian_lr_[0]) + (t*S*M + s*M),
                           &(gmm_lr_[0]) + t*S + s);
      alpha_[t*S + s] = gmm_lr_[t*S + s];
    } else {
      for (unsigned int s = 0; s < S; s++) {
        if (alpha_[(t - 1)*S + s] != LOG_ZERO ||
            (s>0 && alpha_[(t - 1)*S + s - 1] != LOG_ZERO)) {
          if (s == 0) {
            alpha_[t*S + s] = alpha_[(t - 1)*S + s];
          } else {
            alpha_[t*S + s] = idec::LogAdd(alpha_[(t - 1)*S + s],
                                           alpha_[(t - 1)*S + s - 1]);
          }


          // pruning
          if (alpha_beam <0 || alpha_[t*S + s] > max_alpha - alpha_beam) {
            hmm_.states_[s].OutP(feat[t + start], dim, &(gaussian_lr_[0]) + (t*S*M + s*M),
                                 &(gmm_lr_[0]) + t*S + s);
            alpha_[t*S + s] += gmm_lr_[t*S + s];
            max_alpha = std::max(alpha_[t*S + s], max_alpha);
          } else {
            alpha_[t*S + s] = LOG_ZERO;
          }
        }
      }


      // pruning
      if (alpha_beam > 0) {
        for (unsigned int s = 0; s < S; s++) {
          if (alpha_[t*S + s] < max_alpha - alpha_beam) {
            alpha_[t*S + s] = LOG_ZERO;
          }
        }
      }
    } // end T
  }
}

void  HMMTrainer::Forward(const idec::xnnFloatRuntimeMatrix &feat,
                          float alpha_beam, unsigned int start, unsigned int end) {
  unsigned int S = hmm_.NumState();
  unsigned int M = hmm_.states_[0].mixture_.size();
  unsigned int T = (end - start + 1);
  unsigned int dim = feat.NumRows();
  alpha_.assign(T*S, LOG_ZERO);
  gaussian_lr_.assign(T*S*M,
                      LOG_ZERO);  //[T][S][M], [t, s, m] == >t*S*M + s*M + m, maybe out-of-memory?
  gmm_lr_.assign(T*S, LOG_ZERO);

  //[t, s] == >t*S + s
  for (unsigned int t = 0; t < T; t++) {
    double max_alpha = LOG_ZERO;
    if (t == 0) {
      unsigned int s = 0;
      hmm_.states_[s].OutP(feat.Col(t + start), dim,
                           &(gaussian_lr_[0]) + (t*S*M + s*M), &(gmm_lr_[0]) + t*S + s);
      alpha_[t*S + s] = gmm_lr_[t*S + s];
    } else {
      for (unsigned int s = 0; s < S; s++) {
        if (alpha_[(t - 1)*S + s] != LOG_ZERO ||
            (s>0 && alpha_[(t - 1)*S + s - 1] != LOG_ZERO)) {
          if (s == 0) {
            alpha_[t*S + s] = alpha_[(t - 1)*S + s];
          } else {
            alpha_[t*S + s] = idec::LogAdd(alpha_[(t - 1)*S + s],
                                           alpha_[(t - 1)*S + s - 1]);
          }


          // pruning
          if (alpha_beam <0 || alpha_[t*S + s] > max_alpha - alpha_beam) {
            hmm_.states_[s].OutP(feat.Col(t + start), dim,
                                 &(gaussian_lr_[0]) + (t*S*M + s*M), &(gmm_lr_[0]) + t*S + s);
            alpha_[t*S + s] += gmm_lr_[t*S + s];
            max_alpha = std::max(alpha_[t*S + s], max_alpha);
          } else {
            alpha_[t*S + s] = LOG_ZERO;
          }
        }
      }


      // pruning
      if (alpha_beam > 0) {
        for (unsigned int s = 0; s < S; s++) {
          if (alpha_[t*S + s] < max_alpha - alpha_beam) {
            alpha_[t*S + s] = LOG_ZERO;
          }
        }
      }
    } // end T
  }
}

void  HMMTrainer::Backward(float **feat, int dims, int cols, float alpha_beam,
                           float acc_threshold, unsigned int start, unsigned int end) {
  unsigned int S = hmm_.NumState();
  unsigned int M = hmm_.states_[0].mixture_.size();
  unsigned int T = (end - start + 1);
  unsigned int dim = dims;
  beta_.assign(S, LOG_ZERO);

  for (int t = T - 1; t >= 0; t--) {
    if (t == T - 1) {
      unsigned int s = S - 1;
      if (gmm_lr_[t*S + s] == LOG_ZERO) {
        printf("************\n");
      }
      assert(gmm_lr_[t*S + s] != LOG_ZERO);
      beta_[s] = gmm_lr_[t*S + s];
      double gamma = beta_[s] + alpha_[t*S + s] - gmm_lr_[t*S + s] - alpha_[(T - 1)*S
                     + S - 1];
      assert(fabs(gamma) < 0.001);
      states_[s].Accumulate(feat[t+start], dim, exp(gamma),
                            (&(gaussian_lr_[0]) + (t*S*M + s*M)), gmm_lr_[t*S + s], acc_threshold);
    } else {
      beta_prev_ = beta_;
      beta_.assign(S, LOG_ZERO);
      double sum = LOG_ZERO;
      for (int s = S - 1; s >= 0; s--) {
        if (alpha_[t*S + s] != LOG_ZERO) {
          if (beta_prev_[s] != LOG_ZERO ||
              (s != S - 1 && beta_prev_[s + 1] != LOG_ZERO)) {
            if (s == S - 1) {
              beta_[s] = beta_prev_[s];
            } else {
              beta_[s] = idec::LogAdd(beta_prev_[s], beta_prev_[s + 1]);
            }

            if (gmm_lr_[t*S + s] == LOG_ZERO) {
              hmm_.states_[s].OutP(feat[t+start], dim, &(gaussian_lr_[0]) + (t*S*M + s*M),
                                   &(gmm_lr_[0]) + t*S + s);
            }
            beta_[s] += gmm_lr_[t*S + s];

            double gamma = beta_[s] + alpha_[t*S + s] - gmm_lr_[t*S + s] - alpha_[(T - 1)*S
                           + S - 1];
            sum = idec::LogAdd(sum, gamma);
            if (gamma > LOGSMALL_F) {
              states_[s].Accumulate(feat[t+start], dim, exp(gamma),
                                    (&(gaussian_lr_[0]) + (t*S*M + s*M)), gmm_lr_[t*S + s], acc_threshold);
            }
          }
        }
      }

      if (alpha_beam < 0) {
        assert(fabs(sum) < 0.1);
      }
    }
  }
}

void  HMMTrainer::Backward(const idec::xnnFloatRuntimeMatrix &feat,
                           float alpha_beam, float acc_threshold, unsigned int start, unsigned int end) {
  unsigned int S = hmm_.NumState();
  unsigned int M = hmm_.states_[0].mixture_.size();
  unsigned int T = (end - start + 1);
  unsigned int dim = feat.NumRows();
  beta_.assign(S, LOG_ZERO);

  for (int t = T - 1; t >= 0; t--) {
    if (t == T - 1) {
      unsigned int s = S - 1;
      if (gmm_lr_[t*S + s] == LOG_ZERO) {
        printf("************\n");
      }
      assert(gmm_lr_[t*S + s] != LOG_ZERO);
      beta_[s] = gmm_lr_[t*S + s];
      double gamma = beta_[s] + alpha_[t*S + s] - gmm_lr_[t*S + s] - alpha_[(T - 1)*S
                     + S - 1];
      assert(fabs(gamma) < 0.001);
      states_[s].Accumulate(feat.Col(t+start), dim, exp(gamma),
                            (&(gaussian_lr_[0]) + (t*S*M + s*M)), gmm_lr_[t*S + s], acc_threshold);
    } else {
      beta_prev_ = beta_;
      beta_.assign(S, LOG_ZERO);
      double sum = LOG_ZERO;
      for (int s = S - 1; s >= 0; s--) {
        if (alpha_[t*S + s] != LOG_ZERO) {
          if (beta_prev_[s] != LOG_ZERO ||
              (s != S - 1 && beta_prev_[s + 1] != LOG_ZERO)) {
            if (s == S - 1) {
              beta_[s] = beta_prev_[s];
            } else {
              beta_[s] = idec::LogAdd(beta_prev_[s], beta_prev_[s + 1]);
            }

            if (gmm_lr_[t*S + s] == LOG_ZERO) {
              hmm_.states_[s].OutP(feat.Col(t+start), dim,
                                   &(gaussian_lr_[0]) + (t*S*M + s*M), &(gmm_lr_[0]) + t*S + s);
            }
            beta_[s] += gmm_lr_[t*S + s];

            double gamma = beta_[s] + alpha_[t*S + s] - gmm_lr_[t*S + s] - alpha_[(T - 1)*S
                           + S - 1];
            sum = idec::LogAdd(sum, gamma);
            if (gamma > LOGSMALL_F) {
              states_[s].Accumulate(feat.Col(t+start), dim, exp(gamma),
                                    (&(gaussian_lr_[0]) + (t*S*M + s*M)), gmm_lr_[t*S + s], acc_threshold);
            }
          }
        }
      }

      if (alpha_beam < 0) {
        assert(fabs(sum) < 0.1);
      }
    }
  }
}

}
