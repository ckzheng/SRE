#ifndef KALDI_IDEC_COMM_H_
#define KALDI_IDEC_COMM_H_ 1

// idec replacement of kaldi things
#include "base/idec_types.h"
#include "base/idec_common.h"
#include "util/options-itf.h"


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4056 4305 4800 4267 4996 4756 4661)
#define __restrict__
#endif

// Idec replacement of kaldi things
#define  KALDI_ASSERT_IS_INTEGER_TYPE IDEC_ASSERT_IS_INTEGER_TYPE
#define  KALDI_DISALLOW_COPY_AND_ASSIGN IDEC_DISALLOW_COPY_AND_ASSIGN
#define  KALDI_COMPILE_TIME_ASSERT IDEC_COMPILE_TIME_ASSERT

#endif
