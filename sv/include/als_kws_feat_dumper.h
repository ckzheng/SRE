#ifndef INCLUDE_ALS_KWS_FEAT_DUMPER_H_
#define INCLUDE_ALS_KWS_FEAT_DUMPER_H_

#include "als_kws_recognizer.h"

struct AlsKwsAlignResult {
  float score;       // score between 0-100
  float start_time;  // start time of keyword in terms of seconds
  float end_time;    // end time of keyword in terms of seconds
  char *keyword;     // the keyword string
};

// totally for internal usage
class AlsKwsFeatDumper : public AlsKwsRecognizer {
 public:
  virtual ~AlsKwsFeatDumper() {}
  // whether enable the feature dump
  virtual void EnableFeatDump(bool enable) = 0;
  virtual void SetFeatDumpFile(const char *fname) = 0;  // set the dst file name
  // called before BeginUtterance for key
  virtual void SetUtteranceKey(const char *key) = 0;
  virtual void GetAlignResult(AlsKwsAlignResult**result, int *num_word) = 0;
  virtual void FreeAlignResult(AlsKwsAlignResult **, int *num_word) = 0;
};

#endif  // INCLUDE_ALS_KWS_FEAT_DUMPER_H_

