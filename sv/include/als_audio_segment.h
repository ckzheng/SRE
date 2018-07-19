#ifndef __ALS_AUDIO_SEGMENT_H__
#define __ALS_AUDIO_SEGMENT_H__

enum AlsAudioFormat {
    ALS_NULL_AUDIOFORMAT = 0,
    ALS_8K_16BIT_PCM_MONO,
    ALS_8K_8BIT_ALAW_MONO,
    ALS_8K_8BIT_MULAW_MONO,
    ALS_16K_16BIT_PCM_MONO,
    ALS_16K_8BIT_ALAW_MONO,
    ALS_16K_8BIT_MULAW_MONO,
    ALS_16K_4BIT_ADPCM_MONO,
    ALS_RAW_FEATURE_FORMAT=9999
};
struct AlsAudioSegment{
    char*                    buffer;       // the start address of the buffer
    int                      buffer_len;       // the length of the buffer
    AlsAudioFormat           audio_format;  // the audio format of the buffer
};


#endif
