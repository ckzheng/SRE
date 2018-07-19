#ifndef ASR_DECODER_SRC_CORE_VAD_NN_VAD_H_
#define ASR_DECODER_SRC_CORE_VAD_NN_VAD_H_
#include <stdint.h>
#include <deque>
#include <list>
#include <vector>
#include "als_vad.h"
#include "am/xnn_am_scorer.h"
#include "fe/frontend_pipeline.h"
#include "vad/vad_options.h"
#include "util/parse-options.h"
namespace idec {
class FrontendPipeline;
class xnnFloatRuntimeMatrix;
class xnnNet;
class xnnAmEvaluator;
class VADXOptions;
const static float kSilToSpeechTranstionRatio = 0.8f;
const static float kSpeechToSilenceTransitionRatio = 0.1f;
const static int MAX_VOICE_SEGMENTS = 200;
enum VadStateMachine {
  kVadInStateStartPointNotDetected = 1,
  kVadInStateInSpeechSegment = 2,
  kVadInStateEndPointDetected = 3
};
// a speech segment after vad, for internal buffering
struct NNVadSpeechBuf {
  unsigned int start_ms;  // the time span [start_ms, end_ms)
  unsigned int end_ms;
  std::vector<int16_t> buffer;  // the data buffer
  // whether this buffer has the start point triggered
  bool contain_seg_start_point;
  // whether this buffer has the end point event triggered
  bool contain_seg_end_point;
  NNVadSpeechBuf() {
    Reset();
  }
  void Reset() {
    start_ms = end_ms = 0;
    contain_seg_start_point = contain_seg_end_point = false;
    buffer.resize(0);
  }
};
class NNVad : public AlsVad {
 public:
  NNVad(const char *cfg, xnnNet *xnn_net) :po_("vad params initialize") {
    if (NULL == cfg || strlen(cfg) == 0) {
      IDEC_ERROR << "Invalid cfg file! " << cfg << "not exit!\n";
    }
    is_new_api_enable_ = false;
    // initialize frontend
    fe_.Init(cfg, "");
    fe_.BeginUtterance();
    // load vad params
    vad_opts_.Register(&po_);
    po_.ReadConfigFile(cfg);
    // check params
    CheckParams();
    if (xnn_net->vDim() != fe_.GetFeatureDim()) {
      IDEC_ERROR << "mismatched xnnet dim: " << xnn_net->vDim() <<
                 "vs: feature dim: " << fe_.GetFeatureDim();
    }
    // load vad network and initialize vad  inner params
    Init(xnn_net, (int32_t)fe_.FrameShiftInMs());
    xnn_net_ = xnn_net;
    is_final_send_ = false;
    data_buf_start_frame_ = 0;
    frm_cnt_ = 0;
    latest_confirmed_speech_frame_ = 0;
    lastest_confirmed_silence_frame_ = 0;
    continous_silence_frame_count_ = 0;
    vad_state_machine_ = kVadInStateStartPointNotDetected;
    confirmed_start_frame_ = 0;
    confirmed_end_frame_ = 0;
    number_end_time_detected_ = 0;
  }
  virtual ~NNVad() {
    Destroy();
  }
  virtual RetCode Init(xnnNet *xnn_net_, int frame_in_ms);
  virtual void Destroy();
  virtual void Uninit();
  virtual void BeginUtterance();
  virtual void EndUtterance();
  // set the parameters
  virtual int  CheckParams();
  /* hz = [8000,16000] , always 16bit,mono*/
  virtual bool SetSampleRate(int hz);
  virtual int  GetSampleRate();
  virtual bool SetWindowSize(int window_size);
  virtual bool SetEndSilence(int duration);
  virtual bool SetStartSilence(int duration);
  virtual bool SetMaxSpeechTimeout(int duration);
  virtual bool SetSil2SpeechThres(int sil_to_speech_time);
  virtual bool SetSpeech2SilThres(int speech_to_sil_time);
  virtual bool SetStartDetect(bool enable);
  virtual bool SetEndDetect(bool enable);
  virtual bool SetDetectMode(int mode);
  virtual int  GetLatency();
  virtual bool SetLatency(int len);
  // enable the function of detecting voice start position
  virtual void EnableVoiceStartDetect();
  // disable the function of detecting voice start position
  virtual void DisableVoiceStartDetect();
  // enable the function of detecting voice end position
  virtual void EnableVoiceStopDetect();
  // disable the function of detecting voice end position
  virtual void DisableVoiceStopDetect();
  virtual void SetVoiceDetectedCallback(VADCallBack callback, void *param) {
    voice_detected_callback_ = callback;
    voice_detected_param_ = param;
  }
  virtual void SetVoiceStartCallback(VADCallBack callback, void *param) {
    voice_start_callback_ = callback;
    voice_start_param_ = param;
  }
  virtual void SetVoiceEndCallback(VADCallBack callback, void *param) {
    voice_end_callback_ = callback;
    voice_end_param_ = param;
  }
  virtual void SetSilenceDetectedCallback(VADCallBack callback, void *param) {
    silence_detected_callback_ = callback;
    silence_detected_param_ = param;
  }
  virtual bool SetData(int16_t pcm[], int size_in_bytes);

  virtual bool SetParam(const char *name, const char *value);

  virtual int  DoDetect(bool final_frames);
  // new api interface
  virtual bool SetData2(int16_t *pcm, int size_in_bytes, bool final_frame);
  virtual AlsVadResult *DoDetect2();
  // for api wrapper
  AlsVadResult *AllocateApiOutputBuf();  // allocate speech segment
  static void FreeApiOutputBuf(AlsVadResult *&);  // allocate speech segment
  xnnNet *GetModel() { return xnn_net_; }

 public:
  ParseOptions po_;
  VADXOptions vad_opts_;
  // vad result which get from xnneval result
  enum FrameState {
    kFrameStateInvalid = -1,
    kFrameStateSpeech = 1,
    kFrameStateSil = 0
  };
  // final voice/unvoice state per frame
  enum AudioChangeState {
    kChangeStateSpeech2Speech = 0,
    kChangeStateSpeech2Sil = 1,
    kChangeStateSil2Sil = 2,
    kChangeStateSil2Speech = 3,
    kChangeStateNoBegin = 4,
    kChangeStateInvalid = 5
  };

 private:
  void ResetDetection();
  int DetectCommonFrames();
  int DetectLastFrames();
  void DetectOneFrame(FrameState cur_frm_state, int cur_frm_idx,
                      bool is_final_frame);
  void MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int cur_frm_idx);
  int LatencyFrmNumAtStartPoint();
  // state machine class, adjust for vad final result from xnneval result
  class WindowDetector {
   public:
    const static int frameConsThres = 200;
    const static int hyderTimeThres = 80;
    const static int kSmoothWinLenMilliSec = 10;  // ms

   private:
    int *win_state_;
    int cur_win_pos_;
    int win_size_frame_;
    int win_sum_;
    int sil_to_speech_frmcnt_thres_;
    int speech_to_sil_frmcnt_thres_;
    FrameState pre_frame_state_;
    FrameState cur_frame_state_;
    int voice_last_frame_count_;
    int noise_last_frame_count_;
    int hydre_frame_count_;
    int frame_size_ms_;           // frame size in ms, 10 for typical value

   public:
    WindowDetector(int window_size_ms,
                   int sil_to_speech_time,
                   int speech_to_sil_time,
                   int ms_per_frame);
    ~WindowDetector();
    AudioChangeState DetectOneFrame(FrameState frame_state, int frame_count);
    int GetWinSize();
    int FrameSizeMs() { return frame_size_ms_; }
    void Reset();
  };

 private:
  // Constant define
  const static int kVadSampleRate16K = 16000;
  const static int kVadSampleRate8K = 8000;
  const static int kVoiceContinusNum = 7;
  const static int kUnVoiceContinusNum  = 6;
  const static int kFrameShiftMs = 10;  // ms
  // nn evaluator and features
  FrontendPipeline      fe_;
  xnnFloatRuntimeMatrix feat_;
  xnnNet                *xnn_net_;
  XNNAcousticModelScorer *xnn_evaluator_;
  // hang-over module
  WindowDetector *windows_detector_;
  // Callback functions
  VADCallBack voice_detected_callback_;
  VADCallBack silence_detected_callback_;
  VADCallBack voice_start_callback_;
  VADCallBack voice_end_callback_;
  void *voice_detected_param_;
  void *voice_start_param_;
  void *voice_end_param_;
  void *silence_detected_param_;

  void OnSilenceDetected(int frame_idx);
  void OnVoiceDetected(int frame_idx);
  void OnVoiceStart(int frame_idx, bool fake_result = false);
  void OnVoiceEnd(int frame_idx, bool fake_result, bool is_final_frame);

  // stats for make the decision
  int frm_cnt_;                     // total frame of data processed,
  // max value before overflow is about 6,000h
  int latest_confirmed_speech_frame_;   // the last confirmed frame
  int lastest_confirmed_silence_frame_;  // the last confirmed frame
  int continous_silence_frame_count_;   // duration of the continuous non-speech
  // till current frame
  VadStateMachine   vad_state_machine_;  // core state machine
  double noise_average_decibel_;
  int confirmed_start_frame_;
  int confirmed_end_frame_;
  int number_end_time_detected_;
  bool is_final_send_;  // last send. reset per utterance

 private:
  // data buffer related thing
  // called before doing detection
  void ResetOutputBuf();
  // discard the data till frame_idx (including)
  void PopDataBufTillFrame(int frame_idx);
  // pop out the data of [start_frm, start_frm + frm_cnt)
  void PopDataToOutputBuf(int start_frm, int frm_cnt,
                          bool first_frm_is_start_point,
                          bool last_frm_is_end_point,
                          bool end_point_is_last_frame);
  AlsVadResult *CopyOutputBufToApi();
  void NNVadSpeechSegmentToAlsVadSpeechSegment(NNVadSpeechBuf &src,
      AlsVadSpeechBuf &dst);


  bool is_new_api_enable_;
  std::deque<int16_t> data_buf_;  // the data buffer
  uint32 data_buf_start_frame_;
  std::vector <NNVadSpeechBuf> output_data_buf_;
};
}  // namespace idec
#endif  // ASR_DECODER_SRC_CORE_VAD_NN_VAD_H_

