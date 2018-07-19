#ifndef __ALI_ITN_SINGLETON_H__
#define __ALI_ITN_SINGLETON_H__
#include "als_error.h"

class AliItnSingleton {
public:
    static ALSAPI_EXPORT ALS_RETCODE Initialize(const char* far_model_file, const char* grms_file);
    static ALSAPI_EXPORT ALS_RETCODE Initialize(const char* far_model_file, const char* grms_file, const char* crf_model_file);
    static ALSAPI_EXPORT const char* GetVersion();
    static ALSAPI_EXPORT ALS_RETCODE UnInitialize();
};

#endif
