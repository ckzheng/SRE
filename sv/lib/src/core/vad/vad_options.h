#ifndef ASR_DECODER_SRC_CORE_VAD_VAD_OPTIONS_H_
#define ASR_DECODER_SRC_CORE_VAD_VAD_OPTIONS_H_
#include <string>
#include <algorithm>
#include "util/options-itf.h"


namespace idec {
enum VadDetectMode {
  kVadSingleUtteranceDetectMode = 0,
  kVadMutipleUtteranceDetectMode = 1
};
class VADXOptions {
 public:
  int   sample_rate;                // input data sampleRate [8000~16000]hz, always 16bit, mono
  int   detect_mode;                // 0 :single utterance mode  1: multi utterances mode.
  // default 0
  int   max_end_silence_time;       // use for decide how long to end the voice,
  // [0~3000]ms, default 500ms
  int   max_start_silence_time;     // max noise last time at the beginning of input,
  // [0~3000]ms, default 2000ms
  int   max_speech_time;            // max speech segment allowed, default 1min=60,000ms
  bool  do_start_point_detection;   // whether output the start point
  bool  do_end_point_detection;
  int   window_size_ms;            // state machine:smooth window length (ms)
  int   sil_to_speech_time_thres;  // state machine: speech frame more than define in
  // the window, voice starts here (ms)
  int   speech_to_sil_time_thres;  // state machine : silence frame more than define in
  // the window, voice ends here (ms)
  float speech_2_noise_ratio;      // network output speech value / noise value,
  // [1.0 ~ 2.0] default 1
  int   do_extend;                 // whether to extent output timestamp ,0: no extend 1: extend
  int   lookback_time_start_point;  // look back in start point determinization
  int   lookahead_time_end_point;  // look ahead in end point determinization
  int   max_single_segment_time;   // max single segment range , [5s,60s),
  // default 55s, in terms of ms
  bool  is_new_api_enable;         // is new api enable
  int   nn_eval_block_size;        //
  int   dcd_block_size;            // for the block-wise decoding
  float snr_thres_;
  int   noise_frame_num_used_for_snr_;
  float decibel_thres_;
  float speech_noise_thres_;  // speech and noise threshold
  bool  vad_model_use_prior;
  std::string vad_model_path;
  std::string vad_model_format;



  VADXOptions() :
    sample_rate(8000),
    detect_mode(kVadSingleUtteranceDetectMode),
    max_end_silence_time(800),
    max_start_silence_time(3000),
    do_start_point_detection(true),
    do_end_point_detection(true),
    window_size_ms(300),
    sil_to_speech_time_thres(200),
    speech_to_sil_time_thres(150),
    vad_model_path(""),
    vad_model_use_prior(true),
    speech_2_noise_ratio(1.0),
    do_extend(1),
    lookback_time_start_point(120),
    lookahead_time_end_point(120),
    max_single_segment_time(60000),
    is_new_api_enable(true),
    nn_eval_block_size(1),
    dcd_block_size(1),
    snr_thres_(-1000.0f),
    noise_frame_num_used_for_snr_(200),
    decibel_thres_(-100.f), speech_noise_thres_(0.0f) {}


 public:
  bool ToBool(std::string &str, bool *out) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    // allow "" as a valid option for "true",
    // so that --x is the same as --x=true
    if ((str.compare("true") == 0) || (str.compare("t") == 0)
        || (str.compare("1") == 0) || (str.compare("") == 0)) {
      *out = true;
      return true;
    }
    if ((str.compare("false") == 0) || (str.compare("f") == 0)
        || (str.compare("0") == 0)) {
      *out = false;
      return true;
    }
    // if it is neither true nor false:
    return false;  // never reached
  }

  bool ToInt(const std::string &str, int32 *out) {
    char *end_pos;
    // strtol is cheaper than stringstream...
    // strtol accepts decimal 438143, hexa 0x1f2d3 and octal 067123
    int32 ret = std::strtol(str.c_str(), &end_pos, 0);
    if (str.c_str() == end_pos) {
      return false;
    }
    *out = ret;
    return true;
  }

  bool ToUInt(const std::string &str, uint32 *out) {
    char *end_pos;
    // strtol is cheaper than stringstream...
    // strtol accepts decimal 438143, hexa 0x1f2d3 and octal 067123
    uint32 ret = std::strtoul(str.c_str(), &end_pos, 0);
    if (str.c_str() == end_pos) {
      return false;
    }
    *out = ret;
    return true;
  }

  bool ToFloat(const std::string &str, float *out) {
    char *end_pos;
    // strtod is cheaper than stringstream...
    float ret = static_cast<float>(std::strtod(str.c_str(), &end_pos));
    if (str.c_str() == end_pos) {
      return false;
    }
    *out = ret;
    return true;
  }

  bool ToDouble(const std::string &str, double *out) {
    char *end_pos;
    // strtod is cheaper than stringstream...
    double ret = std::strtod(str.c_str(), &end_pos);
    if (str.c_str() == end_pos) {
      return false;
    }
    *out = ret;
    return true;
  }

  bool SetParam(const std::string &key, const std::string &value) {
    if (0 == strcmp(key.c_str(), "speech-noise-thres")) {
      return ToFloat(value, &speech_noise_thres_);
    } else if (0 == strcmp(key.c_str(), "max-end-silence-time")) {
      return ToInt(value, &max_end_silence_time);
    } else if (0 == strcmp(key.c_str(), "max-start-silence-time")) {
      return ToInt(value, &max_start_silence_time);
    } else {
      return false;
    }
  }

  void Register(OptionsItf *po, std::string name = "NNVAD") {
    po->Register(name + "::sample-frequency", &sample_rate,
                 "Waveform data sample frequency (must match the waveform file,"
                 "if specified there)");
    po->Register(name + "::detect-mode", &detect_mode,
                 "detect single utterance mode or multi utterances mode");
    po->Register(name + "::max-end-silence-time", &max_end_silence_time,
                 "silence time last longer than it ,vad detect stop.");
    po->Register(name + "::max-start-silence-time", &max_start_silence_time,
                 "at the beginning silence last longer than it, "
                 "we think it's a mistake operation, vad detect stop");
    po->Register(name + "::voice-start-detec-flag", &do_start_point_detection,
                 "switch that whether detect voice start point");
    po->Register(name + "::voice-end-detec-flag", &do_end_point_detection,
                 "switch that whether detect voice end point");
    po->Register(name + "::window-size", &window_size_ms,
                 "smooth window size of state machine");
    po->Register(name + "::sil-2-speech-time-thres", &sil_to_speech_time_thres,
                 "switch that whether detect voice start point");
    po->Register(name + "::speech-2-sil-time-thres", &speech_to_sil_time_thres,
                 "switch that whether detect voice end point");
    po->Register(name + "::vad-model-path", &vad_model_path,
                 "vad model path to load");
    po->Register(name + "::vad-model-has-prior", &vad_model_use_prior,
                 "judge vad model whether has prior component");
    po->Register(name + "::vad-model-use-prior", &vad_model_use_prior,
                 "whether use prior for vad model");
    po->Register(name + "::vad-model-format", &vad_model_format,
                 "vad model format kaldi_{nnet1|nnet2}");
    po->Register(name + "::speech-2-noise-ratio", &speech_2_noise_ratio,
                 "network output speech value / noise value");
    po->Register(name + "::do-time-extend", &do_extend,
                 "decide whether to extent output timestamp");
    po->Register(name + "::lookback-time-start-point",
                 &lookback_time_start_point, "time extending at start point");
    po->Register(name + "::lookahead-time-end-point", &lookahead_time_end_point,
                 "frame count of extending backward");
    po->Register(name + "::max-single-segment-time", &max_single_segment_time,
                 "max single timestamp range");
    po->Register(name + "::new-api-mode", &is_new_api_enable,
                 "max single timestamp range");
    po->Register(name + "::nn-eval-block-size", &nn_eval_block_size,
                 "NN evalate block size");
    po->Register(name + "::dcd-block_size", &dcd_block_size,
                 "Blocksize for decoding");

    po->Register(name + "::snr-threshold", &snr_thres_,
                 "min snr threshold for speech");
    po->Register(name + "::noise-frame-number-for-snr",
                 &noise_frame_num_used_for_snr_,
                 "noise frame number for snr estimation");
    po->Register(name + "::decibel-threshold", &decibel_thres_,
                 "decibel threshold");
    po->Register(name + "::speech-noise-thres", &speech_noise_thres_,
                 "speech and noise threshold");
  }
};
};  // namespace idec
#endif  // ASR_DECODER_SRC_CORE_VAD_VAD_OPTIONS_H_

