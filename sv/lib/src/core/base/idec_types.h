#ifndef IDEC_BASE_KALDI_TYPES_H_
#define IDEC_BASE_KALDI_TYPES_H_ 1

#include <limits>
#include <climits>
#include <cassert>
#include <cstdlib>
namespace idec {
// TYPEDEFS ..................................................................
#if (IDEC_DOUBLEPRECISION != 0)
typedef double  BaseFloat;
#else
typedef float   BaseFloat;
#endif

// monophone index type
typedef unsigned char   PhoneId;
#define  kPhnIdStrTemrSym  UCHAR_MAX
#define  KMaxPhoneId       (UCHAR_MAX - 1)
#define  MAX_BASEPHONES  255        // maximum # of basephones, the 255 value is reserved for error checking


// ad-hoc functions to use a phonetic context as the key in a hash_map data structure
struct MContextHashFunctions {
  // comparison function (used for matching, comparison for equality)
  bool operator()(const PhoneId *phn_ctx1, const PhoneId *phn_ctx2) const {
    int i = 0;
    for (; phn_ctx1[i] != kPhnIdStrTemrSym; ++i) {
      assert(phn_ctx2[i] != kPhnIdStrTemrSym);
      if (phn_ctx1[i] != phn_ctx2[i]) {
        return false;
      }
    }
    assert(phn_ctx2[i] == kPhnIdStrTemrSym);
    return true;
  }


  // hash function
  size_t operator()(const PhoneId *phn_ctx) const {
    unsigned int iAcc = 0;
    unsigned int iAux = 0;
    for (int i = 0; phn_ctx[i] != kPhnIdStrTemrSym; ++i) {
      if (i <= 3) {
        iAcc <<= (8 * i);
        iAcc += phn_ctx[i];
      } else {
        iAux = phn_ctx[i];
        iAux <<= (8 * (i % 4));
        iAcc ^= iAux;
      }
    }

    return iAcc;
  }
};


// ad-hoc functions to use phone-string as hash keys
struct PhoneStringHashFunctions {
  // comparison function (used for matching, comparison for equality)
  bool operator()(const PhoneId *phn_ctx1, const PhoneId *phn_ctx2) const {
    int i = 0;
    for (; phn_ctx1[i] != kPhnIdStrTemrSym
         && phn_ctx2[i] != kPhnIdStrTemrSym; ++i) {
      if (phn_ctx1[i] != phn_ctx2[i]) {
        return false;
      }
    }
    return (phn_ctx1[i] == phn_ctx2[i]);
  }


  // hash function
  size_t operator()(const PhoneId *phn_ctx) const {
    unsigned int iAcc = 0;
    unsigned int iAux = 0;
    for (int i = 0; phn_ctx[i] != kPhnIdStrTemrSym; ++i) {
      if (i <= 3) {
        iAcc <<= (8 * i);
        iAcc += phn_ctx[i];
      } else {
        iAux = phn_ctx[i];
        iAux <<= (8 * (i % 4));
        iAcc ^= iAux;
      }
    }

    return iAcc;
  }
};

}

#ifdef _MSC_VER
namespace idec {
typedef unsigned char    uint8;
typedef signed char      int8;
typedef unsigned __int16 uint16;
typedef unsigned __int32 uint32;
typedef __int16          int16;
typedef __int32          int32;
typedef __int64          int64;
typedef unsigned __int64 uint64;
typedef float            float32;
typedef double           double64;
}
#include <basetsd.h>
#define ssize_t SSIZE_T

#else
#include <string>
#include <sstream>
namespace std {   // ugly patch for the to_string since gcc low version in Linux

template < typename T > std::string to_string( const T &n ) {
  std::ostringstream stm ;
  stm << n ;
  return stm.str() ;
}
}

// we can do this a different way if some platform
// we find in the future lacks stdint.h
#include <stdint.h>

namespace idec {
typedef uint8_t         uint8;
typedef int8_t          int8;
typedef uint16_t        uint16;
typedef uint32_t        uint32;
typedef uint64_t        uint64;
typedef int16_t         int16;
typedef int32_t         int32;
typedef int64_t         int64;
typedef float           float32;
typedef double          double64;
typedef unsigned char   BYTE;
}  // end namespace idec
#endif

#endif  // IDEC_BASE_KALDI_TYPES_H_
