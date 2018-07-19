#ifndef _DIARIZATION_OPTIONS_H_
#define _DIARIZATION_OPTIONS_H_

#include <string>
#include "util/options-itf.h"
#include "util/dir_utils.h"
#include "util/parse-options.h"

using std::string;

struct FrontEndOptions {
  bool do_vad;
  bool do_cmvn;
  bool verbose_mode;
  //int frame_shift_ms;
  string feat_type;
  string str_input_type;
  string feat_cmvn_file_path;
  string vad_conf_path;
  string vad_mdl_path;
  string fe_conf_path;

  FrontEndOptions() {
    vad_conf_path = "vad.cfg";
    vad_mdl_path = "vad.mdl";
    feat_cmvn_file_path = "cmvn_global.cmvn_global";
    str_input_type = "FE_8K_16BIT_PCM";
    fe_conf_path = "fe.conf";
    verbose_mode = true;
    //frame_shift_ms = 10;
    do_vad = false;
    do_cmvn = false;
  }

  void Register(idec::OptionsItf *po, string prefix = "SpkDiar.") {
    po->Register("input-type", &str_input_type, "input pcm data type.");
    po->Register(prefix + "VadConf", &vad_conf_path, "the path of the VAD config");
    po->Register(prefix + "FeConf", &fe_conf_path, "the path of the VAD config");
    po->Register(prefix + "VadMdl", &vad_mdl_path, "the path of the VAD model");
    po->Register(prefix + "CmvnFile", &feat_cmvn_file_path,
                 "the path of the cmvn file");
    po->Register(prefix + "VerboseLog", &verbose_mode, "Verbose log");
    po->Register(prefix + "DoVad", &do_vad, "do vad before feature extraction");
    po->Register(prefix + "DoCmvn", &do_cmvn,
                 "do global cmvn when feature extraction");
    //po->Register("Waveform2Filterbank::frame-shift", &frame_shift_ms, "frames shift in feature extraction.");
    po->Register("output-type", &feat_type,
                 "feature type in acoustic feature extraction.");
  }

  void FixPath(const char *sys_dir) {
    if (sys_dir == NULL) {
      return;
    }

    vad_conf_path = idec::Path::Combine(string(sys_dir), vad_conf_path);
    vad_mdl_path = idec::Path::Combine(string(sys_dir), vad_mdl_path);
    feat_cmvn_file_path = idec::Path::Combine(string(sys_dir),
                          feat_cmvn_file_path);
  }
};

struct BicOptions {
  bool verbose_mode;
  bool use_bic;
  double alpha;
  double lambda;
  unsigned int win_size;
  unsigned int step_size;

  BicOptions() {
    use_bic = true;
    verbose_mode = true;
    win_size = 200;
    step_size = 10;
    lambda = 1.0;
    alpha = 0.01;
  }

  void Register(idec::OptionsItf *po, string prefix = "SpkDiar.") {
    po->Register(prefix + "UseBicCriterion", &use_bic, "Use BIC Criterion");
    po->Register(prefix + "WinSize", &win_size, "BIC windows size");
    po->Register(prefix + "StepSize", &step_size, "BIC windows step");
    po->Register(prefix + "Lambda", &lambda, "BIC score lambda value");
    po->Register(prefix + "Alpha", &alpha, "BIC score alpha value");
  }
};

struct AhcOptions {
  bool verbose_mode;
  int  class_number;
  int  average_length;
  int  utterance_number;

  AhcOptions() {
    verbose_mode = true;
    class_number = 6;
    average_length = 10;
    utterance_number = 2;
  }

  void Register(idec::OptionsItf *po, string prefix = "SpkDiar.") {
    po->Register(prefix + "ClassNumber", &class_number,
                 "Class number to change clustering Criterion.");
    po->Register(prefix + "AverageLength", &average_length,
                 "Average length to change clustering Criterion.");
    po->Register(prefix + "UtteranceNumber", &utterance_number,
                 "Average length to change clustering Criterion.");
  }
};

struct IvectorExtractOptions {
  bool verbose_mode;
  int min_frames;
  int max_count;
  int ivector_dim;
  int number_gmm_gselect;
  double min_post;
  double acoustic_weight;

  string gmm_mdl_path;
  string iv_mdl_path;
  string lda_mdl_path;
  string plda_mdl_path;
  string ivector_mean_file_path;
  string update_ivector_method;

  IvectorExtractOptions() {
    verbose_mode = true;
    min_frames = 1;
    number_gmm_gselect = 10;
    min_post = 0.025;
    ivector_dim = 400;
    max_count = 0;
    acoustic_weight = 1.0;
    gmm_mdl_path = "final.ubm";
    iv_mdl_path = "final.ie";
    lda_mdl_path = "transform.mat";
    plda_mdl_path = "plda";
    ivector_mean_file_path = "mean.vec";
    update_ivector_method = "update_by_stats";
  }

  void Register(idec::OptionsItf *po, string prefix = "SpkVer.") {
    po->Register(prefix + "UniMdl", &gmm_mdl_path,
                 "the path of the universal model");
    po->Register(prefix + "IvMdl", &iv_mdl_path, "the path of the ivector model");
    po->Register(prefix + "LdaMdl", &lda_mdl_path, "the path of the VAD model");
    po->Register(prefix + "PldaMdl", &plda_mdl_path, "the path of the plda model");
    po->Register(prefix + "IvectorMean", &ivector_mean_file_path,
                 "the path of the ivector mean");
    //po->Register(prefix + "VerboseLog", &verbose_mode, "Verbose log");
    po->Register(prefix + "MinFrames", &min_frames,
                 "min frames used to register or test the speaker model.");
    po->Register(prefix + "MinPost", &min_post,
                 "min posterior probability of gaussian selection");
    po->Register(prefix + "NumberGmmSelect", &number_gmm_gselect,
                 "number of gmm to be select");
    po->Register(prefix + "IvectorDim", &ivector_dim, "dimension of iVector");
    po->Register(prefix + "MaxCount", &max_count,
                 "max frames used to accumulate posterior.");
    po->Register(prefix + "AcousticWeight", &acoustic_weight,
                 "acoustic weight in calculate posterior.");
    po->Register(prefix + "UpdateIvectorMethod", &update_ivector_method,
                 "update ivector method use to update model.");
  }

  void FixPath(const char *sys_dir) {
    if (sys_dir == NULL) {
      return;
    }
    gmm_mdl_path = idec::Path::Combine(string(sys_dir), gmm_mdl_path);
    iv_mdl_path = idec::Path::Combine(string(sys_dir), iv_mdl_path);
    lda_mdl_path = idec::Path::Combine(string(sys_dir), lda_mdl_path);
    plda_mdl_path = idec::Path::Combine(string(sys_dir), plda_mdl_path);
    ivector_mean_file_path = idec::Path::Combine(string(sys_dir),
                             ivector_mean_file_path);
  }
};

struct PldaOptions {
  bool  normalize_length;
  bool  simple_length_norm;
  bool verbose_mode;
  PldaOptions() :normalize_length(true), simple_length_norm(true),
    verbose_mode(true) {
  }

  void Register(idec::OptionsItf *po, string prefix = "SpkVer.") {
    po->Register(prefix + "NormalizeLength", &normalize_length,
                 "If true, do length normalization as part of PLDA (see "
                 "code for details).");
    po->Register(prefix + "SimpleLengthNormalization", &simple_length_norm,
                 "If true, replace the default length normalization by an "
                 "alternative that normalizes the length of the iVectors to "
                 "be equal to the square root of the iVector dimension.");
    //po->Register(prefix + "VerboseLog", &verbose_mode, "Verbose log");
  }
};

struct HmmOptions {
  bool verbose_mode;
  double gamma;
  double epsilon;
  unsigned int iteration_nb;
  unsigned int viterbi_buffer_length;
  string trans_method;
  string ubm_mdl_path;

  HmmOptions() {
    gamma = 0.8;
    epsilon = 5.0;
    iteration_nb = 2;
    verbose_mode = true;
    trans_method = "Equiprob";
    viterbi_buffer_length = 50;
    ubm_mdl_path = "spkdiar.ubm";
  }

  void Register(idec::OptionsItf *po, string prefix = "SpkDiar.") {
    //po->Register(prefix + "VerboseLog", &verbose_mode, "Verbose log");
    po->Register(prefix + "IterationNb", &iteration_nb, "HMM iteration number");
    po->Register(prefix + "Gamma", &gamma, "HMM node transition prob");
    po->Register(prefix + "TransMethod", &trans_method, "HMM transition method");
    po->Register(prefix + "ubm_mdl_path", &trans_method, "HMM transition method");
    po->Register(prefix + "Epsilon", &epsilon, "Viterbi stopping threshold");
    po->Register(prefix + "UniMdl", &ubm_mdl_path,
                 "the path of the universal model");
    po->Register(prefix + "ViterbiBufferLength", &viterbi_buffer_length,
                 "Viterbi buffer length");
  }
};

#endif
