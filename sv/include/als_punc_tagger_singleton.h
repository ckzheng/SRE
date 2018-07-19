#ifndef __ALI_PUNC_TAGGER_SINGLETON_H__
#define __ALI_PUNC_TAGGER_SINGLETON_H__
#include "als_error.h"

namespace AliSpeech {
    class AliPuncTaggerSingleton {
    public:
        static ALSAPI_EXPORT ALS_RETCODE Initialize(const char* sys_dir);
        static ALSAPI_EXPORT const char* GetVersion();
        static ALSAPI_EXPORT ALS_RETCODE UnInitialize();
    };
}

#endif
