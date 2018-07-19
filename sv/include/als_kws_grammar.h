#ifndef __ALI_SPEECH_KWS_GRAMMAR_H__
#define __ALI_SPEECH_KWS_GRAMMAR_H__
#include "als_error.h"

class AlsKwsRecognizer;
class AlsKwsGrammar {
public:
    virtual ~AlsKwsGrammar() {}
    /** factory method to create a grammar
    */
    static ALSAPI_EXPORT ALS_RETCODE Create(AlsKwsGrammar**, AlsKwsRecognizer *rec);

    /** delete itself
    */
    static ALSAPI_EXPORT ALS_RETCODE Destroy(AlsKwsGrammar**);

    /** file_name  -- file name of the grammar file
    */
    virtual ALS_RETCODE Compile(const char *kws_description_file) = 0;
    virtual ALS_RETCODE CompileAlign(const char *kws_description_file) = 0;


    /** Read a compiled grammar from file
    */
    virtual ALS_RETCODE Load(const char *file_prefix) = 0;


    /** Saves a compiled grammar to a file.
    */
    virtual ALS_RETCODE Save(const char *file_prefix) = 0;
};

#endif
