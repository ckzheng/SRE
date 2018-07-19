#ifndef _SPKDIAR_INTERFACE_H_
#define _SPKDIAR_INTERFACE_H_

#include "spkdiar_result.h"

extern "C" {

  void *Init(const char *conf_dir);

  void *CreateInstance(void *handler);

  int DestroyResult(AlsSpkdiarResult *out_result);

  int DestroyInstance(void *inst);

  AlsSpkdiarResult *SpkDiarization(void *inst, char *wave,
                                   unsigned int wave_len);

  int UnInit(void *handler);

}

#endif