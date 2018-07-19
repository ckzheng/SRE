#ifndef INCLUDE_ALS_AEC_PROCESSOR_H_
#define INCLUDE_ALS_AEC_PROCESSOR_H_

#include "als_audio_segment.h"
#include "als_error.h"

class AlsAecProcessor {
 public:
  static ALSAPI_EXPORT ALS_RETCODE Create(AlsAecProcessor **aecp,
    int sample_rate, int num_channels,
    int bit_per_sample);
  static ALSAPI_EXPORT ALS_RETCODE Destroy(AlsAecProcessor **aecp);
  static ALSAPI_EXPORT const char* GetVersion();

  virtual ~AlsAecProcessor() {}

  // speech_segment: original speech recorded from mic.,
  // filtered speech stored in place.
  // reference_segment: playback decoded audio.
  virtual ALS_RETCODE DoAEC(AlsAudioSegment *speech_segment,
    AlsAudioSegment *reference_segment) = 0;
};

#endif  // INCLUDE_ALS_AEC_PROCESSOR_H_
