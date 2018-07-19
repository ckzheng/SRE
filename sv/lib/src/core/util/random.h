#ifndef ASR_DECODER_SRC_CORE_UTIL_RANDOM_H_
#define ASR_DECODER_SRC_CORE_UTIL_RANDOM_H_
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "base/idec_types.h"

namespace idec {
// a thread-safe & controllable random generator
class RandomGenerator {
 public:
  RandomGenerator() {
    rand_seed_ = TimeSeed();
  }

  /***
  *void srand(seed) - seed the random number generator
  *
  *Purpose:
  *       Seeds the random number generator with the int given.  Adapted from the
  *       BASIC random number generator.
  *
  *Entry:
  *       unsigned seed - seed to seed rand # generator with
  *
  *Exit:
  *       None.
  *
  *Exceptions:
  *
  *******************************************************************************/
  int  RandMax() { return 32767; }

  void SetSeed(uint32 seed) {
    rand_seed_ = seed;
  }

  void SetTimeSeed(uint32 seed) {
    rand_seed_ = TimeSeed();
  }

  /***
  *int rand() - returns a random number
  *
  *Purpose:
  *       returns a pseudo-random number 0 through 32767.
  *
  *Entry:
  *       None.
  *
  *Exit:
  *       Returns a pseudo-random number 0 through 32767.
  *
  *Exceptions:
  *
  *******************************************************************************/
  int  Rand() {
    return(((rand_seed_ = rand_seed_ * 214013L
                          + 2531011L) >> 16) & 0x7fff);
  }

 private:
  uint32 TimeSeed(void) {
    time_t now = time(NULL);
    unsigned char *p = (unsigned char *)&now;
    uint32 seed = 0;
    size_t i;

    for (i = 0; i < sizeof now; i++) {
      seed = seed * (UCHAR_MAX + 2U) + p[i];
    }
    return seed;
  }

 private:
  uint32 rand_seed_;
};

}  //  namespace idec

#endif  //  ASR_DECODER_SRC_CORE_UTIL_RANDOM_H_
