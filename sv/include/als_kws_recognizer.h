#ifndef INCLUDE_ALS_KWS_RECOGNIZER_H_
#define INCLUDE_ALS_KWS_RECOGNIZER_H_

#include "als_audio_segment.h"
#include "als_kws_grammar.h"
#include "als_kws_result.h"

class AlsKwsRecognizer {
 public:
  static ALSAPI_EXPORT ALS_RETCODE Create(AlsKwsRecognizer **recognizer,
                                          const char *cfg_file,
                                          const char *sys_dir);
  static ALSAPI_EXPORT ALS_RETCODE Destroy(AlsKwsRecognizer **recognizer);
  // static ALSAPI_EXPORT const char* GetVersion();

  virtual ~AlsKwsRecognizer() {}
  virtual float GetMaxKeywordLengthInSecond() = 0;
  virtual ALS_RETCODE StartUtterance() = 0;
  virtual ALS_RETCODE LoadGrammar(AlsKwsGrammar*grammar) = 0;
  virtual ALS_RETCODE UnLoadGrammar() = 0;
  virtual ALS_RETCODE PutSpeech(AlsAudioSegment *speech_segment) = 0;
  virtual ALS_RETCODE Advance() = 0;
  virtual ALS_RETCODE GetResult(AlsKwsResult **, int* num_result) = 0;
  virtual ALS_RETCODE FreeResult(AlsKwsResult **, int* num_result) = 0;
  virtual ALS_RETCODE EndUtterance() = 0;
  virtual int CurFrame() = 0;
};

#endif  // INCLUDE_ALS_KWS_RECOGNIZER_H_

