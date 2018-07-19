#ifndef INCLUDE_ALS_VAD_H_
#define INCLUDE_ALS_VAD_H_
#include <stdint.h>
#include "als_error.h"
#include "als_vad_result.h"


typedef void(*VADCallBack)(void *, int);



// interface of vad
class AlsVad {
 public:
  enum RetCode {
    kVadSuccess = 0,
    kVadCanNotLoadModel = -1,
    kVadUnsupportedSampleRate = -2,
    kVadBadCfgFile = -3
  };

  // to create vad from a default, global-shared model
  ALSAPI_EXPORT static AlsVad  *Create(const char *cfg, const char *model_path);
  ALSAPI_EXPORT static AlsVad  *Create(const char *cfg);
  ALSAPI_EXPORT static void     Destroy(AlsVad *vad);


  // create the vad from a model explicitly
  ALSAPI_EXPORT static AlsVadMdlHandle LoadModel(const char *cfg,
      const char *model_path);
  ALSAPI_EXPORT static AlsVadMdlHandle LoadModel(const char *cfg);
  ALSAPI_EXPORT static void UnLoadModel(AlsVadMdlHandle);
  ALSAPI_EXPORT static AlsVad  *CreateFromModel(AlsVadMdlHandle);


  // another set of vad api
  virtual void BeginUtterance() = 0;
  virtual void EndUtterance() = 0;

  virtual bool SetData2(int16_t *pcm, int size_in_bytes, bool final_frame) = 0;
  virtual AlsVadResult *DoDetect2() = 0;

  // depreciated vad api
  virtual bool SetData(int16_t *pcm, int size_in_bytes) = 0;
  virtual int  DoDetect(bool final_frame) = 0;

  // extra functions which used to set/get parameters
  virtual int  CheckParams() = 0;
  virtual bool SetSampleRate(int sample_rate) = 0;
  /* hz = [8000,16000] , always 16bit,mono*/
  virtual int  GetSampleRate() = 0;
  virtual bool SetWindowSize(int window_size) = 0;
  virtual bool SetEndSilence(int duration) = 0;
  virtual bool SetStartSilence(int duration) = 0;
  virtual bool SetMaxSpeechTimeout(int duration) = 0;
  virtual bool SetSil2SpeechThres(int sil_to_speech_time) = 0;
  virtual bool SetSpeech2SilThres(int speech_to_sil_time) = 0;
  virtual bool SetStartDetect(bool enable) = 0;
  virtual bool SetEndDetect(bool enable) = 0;
  virtual bool SetDetectMode(int mode) = 0;
  virtual int  GetLatency() = 0;
  virtual bool SetLatency(int latency_ms) = 0;
  virtual void EnableVoiceStartDetect() = 0;
  // enable the function of detecting voice start position
  virtual void DisableVoiceStartDetect() = 0;
  // disable the function of detecting voice start position
  virtual void EnableVoiceStopDetect() = 0;
  // enable the function of detecting voice end position
  virtual void DisableVoiceStopDetect() = 0;
  // disable the function of detecting voice end position
  virtual bool SetParam(const char *name, const char *value) = 0;


  virtual void SetVoiceDetectedCallback(VADCallBack callback, void *param) = 0;
  virtual void SetVoiceStartCallback(VADCallBack callback, void *param) = 0;
  virtual void SetVoiceEndCallback(VADCallBack callback, void *param) = 0;
  virtual void SetSilenceDetectedCallback(VADCallBack callback,
                                          void *param) = 0;

  virtual ~AlsVad() {}
};

#endif  // INCLUDE_ALS_VAD_H_
