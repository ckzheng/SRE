#ifndef INCLUDE_ALS_VAD_RESULT_H_
#define INCLUDE_ALS_VAD_RESULT_H_

#include "als_error.h"

typedef char als_bool;
typedef void *AlsVadMdlHandle;
const AlsVadMdlHandle kInvalidAlsVadHandle = 0;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// a speech segment after vad
struct AlsVadSpeechBuf {
  unsigned int start_ms;  // the time span [start_ms, end_ms)
  unsigned int end_ms;
  void *data;
  // the data, memory managed by api internal
  unsigned int data_len;
  als_bool contain_seg_start_point;
  // whether this segment has the begin point event triggered
  als_bool contain_seg_end_point;
  // whether this segment has the end point event triggered
};

struct AlsVadResult {
  struct AlsVadSpeechBuf *speech_segments;
  int num_segments;
};

ALSAPI_EXPORT void AlsVadResult_Release(struct AlsVadResult **self);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // INCLUDE_ALS_VAD_RESULT_H_

