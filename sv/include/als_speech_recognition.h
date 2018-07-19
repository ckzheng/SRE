#ifndef __ALI_SPEECH_RECOGNITION_H__
#define __ALI_SPEECH_RECOGNITION_H__
#include "als_error.h"

class AliSpeechRecognition {
public:
    static ALSAPI_EXPORT ALS_RETCODE Initialize(const char* cfg_file, const char*sys_dir);
    static ALSAPI_EXPORT const char* GetVersion();
    static ALSAPI_EXPORT ALS_RETCODE UnInitialize();
    static ALSAPI_EXPORT ALS_RETCODE LoadOneDefaultGrammar(const char* grammar_name, const char *grammar_type, const char *file_prefix);
    static ALSAPI_EXPORT ALS_RETCODE LoadDefaultRescoreGrammarXmlScp(const char *file_prefix);
    static ALSAPI_EXPORT bool HasDefaultGrammar(const char* grammar_name, const char *grammar_type);

};

#endif
