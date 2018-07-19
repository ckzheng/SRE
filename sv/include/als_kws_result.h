#ifndef __ALS_KWS_RESULT_H__
#define __ALS_KWS_RESULT_H__

#include "als_error.h"

#define kMainKeywordTag "main_keyword"
#define kOtherKeywordTag "other_keyword"
// tag of the main keyword
struct AlsKwsResult{
    float confidence;  // score between 0-100
    float threshold;   // the threshold it compared to
    float start_time;  // start time of keyword in terms of seconds
    float end_time;    // end time of keyword in terms of seconds
    char *keyword;     // the keyword string
    char *tag;         // general purpose tag
}; 

#endif
