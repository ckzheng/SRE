#ifndef __ALI_PUNC_TAGGER_H__
#define __ALI_PUNC_TAGGER_H__
//#include "als_recognizer_result.h"
#include "als_error.h"
#include <string>

class AlsPuncTagger {
public:
    static ALSAPI_EXPORT ALS_RETCODE Create(AlsPuncTagger **punc);
    virtual ALS_RETCODE Destroy();

    virtual ~AlsPuncTagger() {};
    virtual ALS_RETCODE PuncTagging(const char*, char *, bool is_end = false) = 0;
    virtual ALS_RETCODE PuncTagging(const std::string& input, std::string& output, bool is_end = false) = 0;
    virtual ALS_RETCODE PuncTaggingWithMarkup(const char*, char *, const char*, bool is_end = false) = 0;
    virtual ALS_RETCODE PuncTaggingWithMarkup(const std::string& input, std::string& output, const std::string &str_ignore, bool is_end = false) = 0;
};

#endif
