#ifndef ALS_GEN_LM_H_
#define ALS_GEN_LM_H_
#include "als_error.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
    typedef void*  AlsBiasHandle;
    ALSAPI_EXPORT ALS_RETCODE AlsBiasLmCreate(const char *sys_dir, const char *cfg_file, AlsBiasHandle *hd);
    ALSAPI_EXPORT ALS_RETCODE AlsBiasLmCompile(AlsBiasHandle hd, const char *in_buf, int in_buf_len, char **compiled_lm, int *compiled_lm_len);
    ALSAPI_EXPORT ALS_RETCODE AlsBiasDestroy(AlsBiasHandle *hd);
    ALSAPI_EXPORT ALS_RETCODE AlsBiasFree(AlsBiasHandle *hd, char **data);
};

#endif
