#ifndef GLOBAL_H
#define GLOBAL_H

// open-mp
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#undef min
#undef max
#include <float.h>
#define finite _finite
#endif

// verbose output
// #define IDEC_VERBOSE_ENABLED

// asserts
//#define NDEBUG
#include <cassert>
#include <cmath>
#include <cfloat>
#include <limits>
#include <cmath>
#include <malloc.h>
#include <cstdio>
#include <cerrno>
#include <stdint.h>
#include <stdexcept>


#if defined __linux__ || __MINGW32__
#if __cplusplus < 199711L
#include <map>
#include <set>
#endif
#if __cplusplus < 201103L
#include <tr1/unordered_map>
#include <tr1/unordered_set>
using std::tr1::unordered_map;
using std::tr1::unordered_set;
#define __INCLUDE_TR1__
#else
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#endif
#elif defined __APPLE__
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#elif _MSC_VER
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#else
#error "unsupported platform"
#endif

//
// #if defined __linux__ || defined __APPLE__ || __MINGW32__
// #if __cplusplus < 201103L
// #include "util/shared_ptr.h"
// #include "util/unique_ptr.h"
// // using std::tr1::unordered_map;
// // using std::tr1::unordered_set;
// #else
// #include <memory>
// using std::shared_ptr;
// using std::unique_ptr;
// #endif
// #elif _MSC_VER
// #include <memory>
// using std::shared_ptr;
// using std::unique_ptr;
// #else
// #error "unsupported platform"
// #endif
//


#if defined(_MSC_VER)
#pragma warning(disable: 4996)
#define __restrict__
#endif

namespace idec {


// formatting macros
#define FLT(width,precision) std::setw(width) << std::setiosflags(ios::fixed) << std::setprecision(precision)

// inline function
#if defined __linux__ || defined __APPLE__ || __MINGW32__
#define FORCE_INLINE __attribute__((always_inline))
#define NO_INLINE __attribute__((noinline))
#elif _MSC_VER
#define FORCE_INLINE __forceinline
#define NO_INLINE __declspec(noinline)
#else
#warning "unsupported platform"
#define FORCE_INLINE inline
#define NO_INLINE
#endif



#ifdef HAVE_POSIX_MEMALIGN
#  define IDEC_MEMALIGN(align, size, pp_orig) \
     (!posix_memalign(pp_orig, align, size) ? *(pp_orig) : NULL)
#  define IDEC_MEMALIGN_FREE(x) free(x)
#elif defined(HAVE_MEMALIGN)
/* Some systems have memalign() but no declaration for it */
void *memalign(size_t align, size_t size);
#  define IDEC_MEMALIGN(align, size, pp_orig) \
     (*(pp_orig) = ::memalign(align, size))
#  define IDEC_MEMALIGN_FREE(x) free(x)
#elif defined(_MSC_VER)
#  define IDEC_MEMALIGN(align, size, pp_orig) \
  (*(pp_orig) = _aligned_malloc(size, align))
#  define IDEC_MEMALIGN_FREE(x) _aligned_free(x)
#else
// #error Manual memory alignment is no longer supported
#endif


#ifdef _MSC_VER
#define IDEC_STRCASECMP _stricmp
#else
#define IDEC_STRCASECMP strcasecmp
#endif
#ifdef _MSC_VER
#  define IDEC_STRTOLL(cur_cstr, end_cstr) _strtoi64(cur_cstr, end_cstr, 10);
#else
#  define IDEC_STRTOLL(cur_cstr, end_cstr) strtoll(cur_cstr, end_cstr, 10);
#endif
#define IDEC_STRTOD(cur_cstr, end_cstr) strtod(cur_cstr, end_cstr)

inline void IDEC_ASSERT(bool cond) {
  if (!cond) {
    throw std::runtime_error("");
  }
}

inline void IDEC_ASSERT_DEBUG(bool cond) {
#if _DEBUG
  IDEC_ASSERT(cond);
#endif
}



// Makes copy constructor and operator= private.  Same as in compat.h of OpenFst
// toolkit.  If using VS, for which this results in compilation errors, we
// do it differently.

#if defined(_MSC_VER)
#define IDEC_DISALLOW_COPY_AND_ASSIGN(type) \
  void operator = (const type&)
#else
#define IDEC_DISALLOW_COPY_AND_ASSIGN(type)    \
  type(const type&);                  \
  void operator = (const type&)
#endif

template<bool B> class IdecCompileTimeAssert {};
template<> class IdecCompileTimeAssert < true > {
 public:
  static inline void Check() {}
};

#define IDEC_COMPILE_TIME_ASSERT(b) IdecCompileTimeAssert<(b)>::Check()

#define IDEC_ASSERT_IS_INTEGER_TYPE(I) \
  IdecCompileTimeAssert<std::numeric_limits<I>::is_specialized \
                 && std::numeric_limits<I>::is_integer>::Check()

#define IDEC_ASSERT_IS_FLOATING_TYPE(F) \
  IdecCompileTimeAssert<std::numeric_limits<F>::is_specialized \
                && !std::numeric_limits<F>::is_integer>::Check()

// log s
#define LOG_2_PI	 1.83787706640934548355
#define LOG_10       2.30258509299404568    /* Defined to save recalculating it */
//    #define M_LN10 2.30258509299404568402
#define LOG_ZERO     -3.402823466e+38F
#define MINMIX       1e-5
#define LOG_MINMIX   (-11.5129254649702)
#define LOGSMALL_F	 (-4.5e+6)
#define LOG_ONE 0.0f

#define M_2PI 6.283185307179586476925286766559005

#ifdef _MSC_VER
#define M_PI 3.1415926535897932384626433832795
#define M_SQRT1_2  0.707106781186547524401f
#endif


template<typename  T>
inline void IDEC_DELETE(T &ptr) {
  if (ptr != NULL) {
    delete ptr;
  }
  ptr = NULL;
}


template<typename  T>
inline void IDEC_DELETE_ARRAY(T &ptr) {
  if (ptr != NULL) {
    delete[]ptr;
  }
  ptr = NULL;
}


#ifndef _MSC_VER
#include <unistd.h>
typedef int errno_t;
inline errno_t fopen_s(FILE **f, const char *name, const char *mode) {
  errno_t ret = 0;
  assert(f);
  *f = fopen(name, mode);
  /* Can't be sure about 1-to-1 mapping of errno and MS' errno_t */
  if (!*f)
    ret = errno;
  return ret;
}
#endif


inline float LogAdd(float x, float y) {
  static const float kMinLogDiffFloat = std::log(FLT_EPSILON);  // negative!
  float diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffFloat) {
    float res;
#ifdef _MSC_VER
    res = x + logf(1.0f + expf(diff));
#else
    res = x + log1pf(expf(diff));
#endif
    return res;
  } else {
    return x;  // return the larger one.
  }
}

inline bool ApproximatelyEqnZero(float value) {
  return fabs(value) < FLT_EPSILON * 5;
}
}
#endif
