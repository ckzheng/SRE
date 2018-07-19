#ifndef __ALI_ITN_H__
#define __ALI_ITN_H__
//#include "als_recognizer_result.h"
#include "als_error.h"
#include <string>

class AlsItn {
public:
    static ALSAPI_EXPORT ALS_RETCODE Create(AlsItn **itn);
    virtual ALS_RETCODE Destroy();

    virtual ~AlsItn() {};
    virtual ALS_RETCODE toITN(const char*, const char*, char *) = 0;
    virtual ALS_RETCODE toITN(const std::string& rule, const std::string& input, std::string& output) = 0;
//    virtual ALS_RETCODE toITN(AlsRecognizedPhrase *) = 0;
//    virtual ALS_RETCODE toITN(AlsRecognizerResult *) = 0;
};

#endif
