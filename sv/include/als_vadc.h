#ifndef VAD_WRAPPER_H_
#define VAD_WRAPPER_H_

#include "als_vad_result.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef void*  AlsVadHandle;
    ALSAPI_EXPORT  AlsVadHandle AlsVadCreate(const char *cfg);
    ALSAPI_EXPORT  AlsVadHandle AlsVadCreate2(const char *cfg, const char *model_path);

    ALSAPI_EXPORT  AlsVadMdlHandle AlsVadLoadModel(const char *cfg);
    ALSAPI_EXPORT  AlsVadMdlHandle AlsVadLoadModelFromPath(const char *cfg, const char *model_path);
    ALSAPI_EXPORT  void            AlsVadUnLoadModel(AlsVadMdlHandle);
    ALSAPI_EXPORT  AlsVadHandle    AlsVadCreateFromModel(AlsVadMdlHandle);

    ALSAPI_EXPORT  void         AlsVadDestroy(AlsVadHandle vad);
    ALSAPI_EXPORT  void         AlsVadBeginUtterance(AlsVadHandle hd);
    ALSAPI_EXPORT  void         AlsVadEndUtterance(AlsVadHandle hd);
    ALSAPI_EXPORT  void         AlsVadSetData2(AlsVadHandle hd, short *pcm, int size_in_bytes, als_bool final_frame);
    ALSAPI_EXPORT  struct AlsVadResult *AlsVadDoDetect2(AlsVadHandle hd);

    ALSAPI_EXPORT  void AlsVadSetSampleRate(AlsVadHandle hd, int sample_rate);
    ALSAPI_EXPORT  int  AlsVadGetSampleRate(AlsVadHandle hd);
    ALSAPI_EXPORT  void AlsVadSetEndSilence(AlsVadHandle hd, int duration);
    ALSAPI_EXPORT  void AlsVadSetStartSilence(AlsVadHandle hd, int duration);
    ALSAPI_EXPORT  void AlsVadSetMaxSpeechTimeout(AlsVadHandle hd, int duration);
    ALSAPI_EXPORT  void AlsVadSetDetectMode(AlsVadHandle hd, int mode);
    
#ifdef __cplusplus
}
#endif
#endif  // VAD_WRAPPER_H_
