#ifndef _SPKDIAR_OUT_H_
#define _SPKDIAR_OUT_H_

struct  SpeechFragment {
  unsigned int begin_time;
  unsigned int end_time;
  unsigned int speaker_id;
};

struct  AlsSpkdiarResult {
  SpeechFragment *speech_fragments;
  unsigned int fragment_num;
};

#endif
