#ifndef _ALS_SID_GMM_H
#define _ALS_SID_GMM_H

#include <vector>
#include <iostream>
#include "am/xnn_feature.h"
#include "am/xnn_net.h"
#include "fe/frontend_pipeline.h"
#include "base/serialize_helper.h"
#include "base/log_message.h"
#include "util/parse-options.h"
#include "util/dir_utils.h"
#include "util/text-utils.h"
#include "als_error.h"

using std::vector;

namespace alsid {
#define  MIX_FLOOR  (2.0f*MINMIX)
#define  VAR_FLOOR  0.0001f
struct MapUpdateOption {
  bool update_mean;
  bool update_var;
  bool update_weight;
  float tau_mean;
  float tau_var;
  float tau_weight;
  MapUpdateOption() {
    update_mean = true;
    update_var = true;
    update_weight = true;
    tau_mean = 10.0;
    tau_var = 10.0;
    tau_weight = 10.0;
  }

  void Register(idec::OptionsItf *po, std::string prefix = "SV.SD.") {
    po->Register(prefix + "UpdateM", &update_mean,
                 "number of states for sd model.");
    po->Register(prefix + "UpdateV", &update_var, "the sd model directory.");
    po->Register(prefix + "UpdateW", &update_weight, "the path of the si model");
    po->Register(prefix + "TauM", &tau_mean, "number of states for sd model.");
    po->Register(prefix + "TauV", &tau_var, "the sd model directory.");
    po->Register(prefix + "TauW", &tau_weight, "the path of the si model");
  }
};

class GaussianComponent {
 protected:
  std::vector<float> mean_;
  std::vector<float> var_;
  float gconst_;
 public:
  GaussianComponent() {};
  virtual ~GaussianComponent() {};

  std::vector<float> &Mean() { return mean_; };
  std::vector<float> &Var() { return var_; };
  void SetGconst(float gconst) { gconst_ = gconst; }
  void SetMean(std::vector<float> mean) { this->mean_ = mean; };
  void SetVar(std::vector<float> var) { this->var_ = var; };
  float OutP(const float *x, const unsigned int &dim);
  void  Serialize(idec::SerializeHelper &helper);
  void  Deserialize(idec::SerializeHelper &helper);
  void  FixGConst(); //  when should call this?
 protected:
  void ApplyVarFloor(const std::vector<float> &var_floor);
};

class GaussianTrainer {
 public:
  GaussianTrainer();
  void  Reset(GaussianComponent &gaussian);
  void  Accumulate(GaussianComponent &gaussian, const float *x, unsigned int dim,
                   const double p);
  void  MapUpdate(GaussianComponent &gaussian, const MapUpdateOption &opt);
  void  MapUpdate2(GaussianComponent &gaussian, double weight, const MapUpdateOption &opt);
  void  MleUpdate(GaussianComponent &gaussian, const MapUpdateOption &opt);
  double gamma_;
  double num_frame_;
  std::vector<double> gamma_x_;
  std::vector<double> gamma_xx_;
};

class GMM {
 public:
  void  OutP(const float *x, const unsigned int &dim, float *likelihood_per_mix,
             float *likelihood);
  float OutP(const float *x, const unsigned int &dim);
  void  Serialize(idec::SerializeHelper &helper);
  void  Deserialize(idec::SerializeHelper &helper);
  IDEC_RETCODE  Deserialize(const char *fname);
  unsigned int NumMix() { return  mixture_.size(); }
  vector<float> &getMean(unsigned int i);
  vector<float> &getVar(unsigned int i);
  void SetWeight(vector<float> wt) {wt_ = wt;}
  void SetLWeight(vector<float> wt_log) { wt_log_ = wt_log; }
  void AddGaussianComponent(GaussianComponent gaussian) { mixture_.push_back(gaussian); }
  std::vector<float> wt_;
  std::vector<float> wt_log_;
  std::vector<GaussianComponent> mixture_;
  // observation caches
  std::vector<float> likelihood_;
 private:
  void FixGconst();
};


class GMMTrainer {
 public:
  GMMTrainer();
  void  SetMdl(GMM *gmm);
  void  Reset();
  void  Accumulate(const float *x, unsigned int dim, const double p,
                   float *likelihood_per_mix, const float &likelihood, float acc_threshold);
  void  MapUpdate(const MapUpdateOption &opt);
  void  MapUpdate2(const MapUpdateOption &opt);
  void  MleUpdate(const MapUpdateOption &opt);

 protected:
  std::vector<GaussianTrainer> gmm_trainer_;
  GMM *gmm_;
  std::vector<float> var_floor_;
  std::vector<float> posterior_;
};

class HMM {
 public:
  HMM();
  unsigned int NumState() const { return states_.size(); }
  void   Serialize(const char *fname);
  void   Serialize(idec::SerializeHelper &helper);
  IDEC_RETCODE   Deserialize(const char *fname);
  void   Deserialize(idec::SerializeHelper &helper);
  void   InitFromUbm(const GMM &ubm, const unsigned int &nstate);

  // add by zhuozhu
  //void   AddState(GMM& gmm);
  const vector<GMM>& GetStates() const;

  void AddState(GMM &gmm) { states_.push_back(gmm); };

  // viterbi score
  void   BeginEvaluate(float beam);
  void   Evaluate(const idec::xnnFloatRuntimeMatrix &feat, float *tol_lr);
  void   Evaluate(float **feat, int dims, int cols, float *tol_lr,
                  unsigned int start, unsigned int end);
  void   Evaluate(const idec::xnnFloatRuntimeMatrix &feat, float *tol_lr,
                  std::vector< std::vector<int> > &trace_,
                  std::vector< std::vector<float> > &loglikelihood_);
  void   Evaluate(const idec::xnnFloatRuntimeMatrix &feat, float *tol_lr,
                  unsigned int start, unsigned int end);
  float  EndEvaluate() {};

  std::vector<GMM> states_;
 private:
  std::vector<float> viterbi_score_;      // S
  std::vector<float> viterbi_score_prev_; // S
  unsigned int             decode_cur_frm_;
  float              viterbi_prune_beam_;
};

class HMMTrainer {
 public:
  HMMTrainer(HMM &hmm);
  void  Accumulate(float **feat, int dims, int cols, float alpha_beam,
                   float acc_threshold, unsigned int start, unsigned int end);
  void  Accumulate(const idec::xnnFloatRuntimeMatrix &feat, float alpha_beam,
                   float acc_threshold);
  void  Accumulate(const idec::xnnFloatRuntimeMatrix &feat, float alpha_beam,
                   float acc_threshold, unsigned int start, unsigned int end);
  void  AccumulateEqualLength(const idec::xnnFloatRuntimeMatrix &feat,
                              float acc_threshold);
  void  MapUpdate(const MapUpdateOption &opt);
  void  MleUpdate(const MapUpdateOption &opt);
  void  Reset();
 private:
  void  Forward(float **feat, int dims, int cols, float alpha_beam,
                unsigned int start, unsigned int end);
  void  Backward(float **feat, int dims, int cols, float alpha_beam,
                 float acc_threshold, unsigned int start, unsigned int end);
  void  Forward(const idec::xnnFloatRuntimeMatrix &feat, float alpha_beam);
  void  Backward(const idec::xnnFloatRuntimeMatrix &feat, float alpha_beam,
                 float acc_threshold);
  void  Forward(const idec::xnnFloatRuntimeMatrix &feat, float alpha_beam,
                unsigned int start, unsigned int end);
  void  Backward(const idec::xnnFloatRuntimeMatrix &feat, float alpha_beam,
                 float acc_threshold, unsigned int start, unsigned int end);
 private:
  std::vector<GMMTrainer> states_;
  std::vector<double> alpha_;       // [T][S] ,  [t,s]==>t*S+s
  std::vector<double> beta_;        // [S]
  std::vector<double> beta_prev_;   // [S]
  std::vector<float>  gaussian_lr_; // [T][S][M], [t,s,m]==>t*S*M+s*M+m
  std::vector<float>  gmm_lr_;      // [T][S] ,  [t,s]==>t*S+s
  HMM &hmm_;
};
}
#endif
