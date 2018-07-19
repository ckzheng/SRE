#ifndef _SPEAKER_VERIFICATION_OPTIONS_H_
#define _SPEAKER_VERIFICATION_OPTIONS_H_

#include <string>
#include "util/options-itf.h"
#include "util/dir_utils.h"
#include "util/parse-options.h"

using std::string;

struct PldaOptions {
  bool  normalize_length;
  bool  simple_length_norm;
  bool verbose_mode;
  PldaOptions():normalize_length(true), simple_length_norm(true),
    verbose_mode(true) {}
  void Register(idec::OptionsItf *po, string prefix = "SpkVer.") {
    po->Register(prefix + "NormalizeLength", &normalize_length,
                 "If true, do length normalization as part of PLDA (see "
                 "code for details).");
    po->Register(prefix + "SimpleLengthNormalization", &simple_length_norm,
                 "If true, replace the default length normalization by an "
                 "alternative that normalizes the length of the iVectors to "
                 "be equal to the square root of the iVector dimension.");
    po->Register(prefix + "VerboseLog", &verbose_mode, "Verbose log");
  }
};

struct IvectorExtractOptions {
  string ubm_mdl_path;
  string vad_conf_path;
  string vad_mdl_path;
  string iv_mdl_path;
  string lda_mdl_path;
  string plda_mdl_path;
  string feat_cmvn_file_path;
  string ivector_mean_file_path;
  string str_input_type;
  string update_ivector_method;
  string feat_type;
  bool verbose_mode;
  int number_gmm_gselect;
  int min_frames;
  double min_post;
  int ivector_dim;
  int max_count;
  int frame_shift_ms;
  int frame_length_ms;
  double acoustic_weight;
  bool do_vad;
  bool do_cmvn;
  float energy_threshold;

  IvectorExtractOptions() {
    ubm_mdl_path = "final.ubm";
    iv_mdl_path = "final.ie";
    lda_mdl_path = "transform.mat";
    plda_mdl_path = "plda";
    vad_conf_path = "vad.cfg";
    vad_mdl_path = "vad.mdl";
    feat_cmvn_file_path = "cmvn_global.cmvn_global";
    ivector_mean_file_path = "mean.vec";
    str_input_type = "FE_8K_16BIT_PCM";
    update_ivector_method = "update_by_stats";
    verbose_mode = true;
    min_frames = 1;
    number_gmm_gselect = 10;
    min_post = 0.025;
    ivector_dim = 400;
    max_count = 0;
    acoustic_weight = 1.0;
    frame_shift_ms = 10;
    frame_length_ms = 25;
    do_vad = false;
    do_cmvn = false;
    energy_threshold = 0.0;
  }

  void Register(idec::OptionsItf *po, string prefix = "SpkVer.") {
    po->Register(prefix + "StrInputType", &str_input_type, "input pcm data type.");
    po->Register(prefix + "UniMdl", &ubm_mdl_path,
                 "the path of the universal model");
    po->Register(prefix + "IvMdl", &iv_mdl_path, "the path of the ivector model");
    po->Register(prefix + "LdaMdl", &lda_mdl_path, "the path of the VAD model");
    po->Register(prefix + "PldaMdl", &plda_mdl_path, "the path of the plda model");
    po->Register(prefix + "VadConf", &vad_conf_path, "the path of the VAD config");
    po->Register(prefix + "VadMdl", &vad_mdl_path, "the path of the VAD model");
    po->Register(prefix + "CmvnFile", &feat_cmvn_file_path,
                 "the path of the cmvn file");
    po->Register(prefix + "IvectorMean", &ivector_mean_file_path,
                 "the path of the ivector mean");
    po->Register(prefix + "VerboseLog", &verbose_mode, "Verbose log");
    po->Register(prefix + "MinFrames", &min_frames,
                 "min frames used to register or test the speaker model.");
    po->Register(prefix + "MinPost", &min_post,
                 "min posterior probability of gaussian selection");
    po->Register(prefix + "NumberGmmSelect", &number_gmm_gselect,
                 "number of gmm to be select");
    po->Register(prefix + "DoVad", &do_vad, "do vad before feature extraction");
    po->Register(prefix + "DoCmvn", &do_cmvn,
                 "do global cmvn when feature extraction");
    po->Register(prefix + "IvectorDim", &ivector_dim, "dimension of iVector");
    // po->Register(prefix + "UseWeights", &use_weights,
    // "if true, regress the log-weights on the iVector");
    po->Register(prefix + "MaxCount", &max_count,
                 "max frames used to accumulate posterior.");
    po->Register("Waveform2Filterbank::frame-shift", &frame_shift_ms,
                 "frames shift in feature extraction.");
    po->Register("Waveform2Filterbank::frame-length", &frame_length_ms,
                 "frames length in feature extraction.");
    po->Register(prefix + "AcousticWeight", &acoustic_weight,
                 "acoustic weight in calculate posterior.");
    po->Register(prefix + "UpdateIvectorMethod", &update_ivector_method,
                 "update ivector method use to update model.");
    po->Register("output-type", &feat_type,
                 "feature type in acoustic feature extraction.");
    po->Register("energy_threshold", &energy_threshold,
                 "time domain energy threshold to remove absolute silence.");
  }

  void FixPath(const char *sys_dir) {
    if (sys_dir == NULL) {
      return;
    }
    ubm_mdl_path = idec::Path::Combine(string(sys_dir), ubm_mdl_path);
    vad_conf_path = idec::Path::Combine(string(sys_dir), vad_conf_path);
    vad_mdl_path = idec::Path::Combine(string(sys_dir), vad_mdl_path);
    iv_mdl_path = idec::Path::Combine(string(sys_dir), iv_mdl_path);
    lda_mdl_path = idec::Path::Combine(string(sys_dir), lda_mdl_path);
    plda_mdl_path = idec::Path::Combine(string(sys_dir), plda_mdl_path);
    feat_cmvn_file_path = idec::Path::Combine(string(sys_dir),
                          feat_cmvn_file_path);
    ivector_mean_file_path = idec::Path::Combine(string(sys_dir),
                             ivector_mean_file_path);
  }
};

#endif

