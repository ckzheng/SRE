#ifndef __ALS_SIG_PROCESS_H__
#define __ALS_SIG_PROCESS_H__

#include "als_error.h"


class AlsSigProcess {
public:
    AlsSigProcess();
    virtual ~AlsSigProcess();
    static ALSAPI_EXPORT ALS_RETCODE Create(AlsSigProcess **sigp, int sample_rate, int num_channels, int bit_per_sample);
    virtual ALS_RETCODE Destroy();


    /* DoAEC:   function for adaptive echo cancellation (AEC)
    in_buf :    pointer of recorded mic signal, managed by user
    spk_buf:    pointer of audio data which will render to speaker, managed by user
    out_buf:    pointer of echo cancelled mic signal, managed by user
    in_buf_len: length of in_buf in bytes.
    out_buf_len length of out_buf in bytes, should be >= in_buf_len.
    */
    ALS_RETCODE DoAEC(char *in_buf, char *spk_buf, int in_buf_len, char *out_buf, int out_buf_len, int &output_len);

#endif