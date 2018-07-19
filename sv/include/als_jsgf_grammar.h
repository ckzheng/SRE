#ifndef ALS_JSGF_GRAMMAR_H_
#define ALS_JSGF_GRAMMAR_H_
#include "als_error.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
    typedef void*  AlsJsgfHandle;
    ALSAPI_EXPORT ALS_RETCODE AlsJsgfCreate(AlsJsgfHandle   *hd);
    ALSAPI_EXPORT ALS_RETCODE AlsJsgfDestroy(AlsJsgfHandle  *hd);
    ALSAPI_EXPORT ALS_RETCODE AlsJsgfCompileFromString(AlsJsgfHandle hd, const char *jsgf_str);
    ALSAPI_EXPORT ALS_RETCODE AlsJsgfGetBinaryImage(AlsJsgfHandle hd, char **data, size_t *data_len);
    ALSAPI_EXPORT ALS_RETCODE AlsJsgfFreeBinaryImage(AlsJsgfHandle hd, char **data);
};

#endif
