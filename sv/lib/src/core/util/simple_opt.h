#ifndef INCLUDED_SimpleOpt
#define INCLUDED_SimpleOpt

// Default the max arguments to a fixed value. If you want to be able to
// handle any number of arguments, then predefine this to 0 and it will
// use an internal dynamically allocated buffer instead.
#ifdef SO_MAX_ARGS
# define SO_STATICBUF   SO_MAX_ARGS
#else
# include <stdlib.h>    // malloc, free
# include <string.h>    // memcpy
# define SO_STATICBUF   50
#endif

typedef enum _ESOError {
  SO_SUCCESS          =  0,

  SO_OPT_INVALID      = -1,

  SO_OPT_MULTIPLE     = -2,

  SO_ARG_INVALID      = -3,

  SO_ARG_INVALID_TYPE = -4,

  SO_ARG_MISSING      = -5,

  SO_ARG_INVALID_DATA = -6
} ESOError;

enum _ESOFlags {
  SO_O_EXACT       = 0x0001,

  SO_O_NOSLASH     = 0x0002,

  SO_O_SHORTARG    = 0x0004,

  SO_O_CLUMP       = 0x0008,

  SO_O_USEALL      = 0x0010,

  SO_O_NOERR       = 0x0020,

  SO_O_PEDANTIC    = 0x0040,

  SO_O_ICASE_SHORT = 0x0100,

  SO_O_ICASE_LONG  = 0x0200,

  SO_O_ICASE_WORD  = 0x0400,

  SO_O_ICASE       = 0x0700
};

typedef enum _ESOArgType {
  SO_NONE,

  SO_REQ_SEP,

  SO_REQ_CMB,

  SO_OPT,

  SO_MULTI
} ESOArgType;

#define SO_END_OF_OPTIONS   { -1, NULL, SO_NONE }

#ifdef _DEBUG
# ifdef _MSC_VER
#  include <crtdbg.h>
#  define SO_ASSERT(b)  _ASSERTE(b)
# else
#  include <assert.h>
#  define SO_ASSERT(b)  assert(b)
# endif
#else
# define SO_ASSERT(b)
#endif

// ---------------------------------------------------------------------------
//                              MAIN TEMPLATE CLASS
// ---------------------------------------------------------------------------

template<class SOCHAR>
class CSimpleOptTempl {
 public:
  struct SOption {
    int nId;

    const SOCHAR *pszArg;

    ESOArgType nArgType;
  };

  CSimpleOptTempl()
    : m_rgShuffleBuf(NULL) {
    Init(0, NULL, NULL, 0);
  }

  CSimpleOptTempl(
    int             argc,
    SOCHAR         *argv[],
    const SOption *a_rgOptions,
    int             a_nFlags = 0
  )
    : m_rgShuffleBuf(NULL) {
    Init(argc, argv, a_rgOptions, a_nFlags);
  }

#ifndef SO_MAX_ARGS

  ~CSimpleOptTempl() { if (m_rgShuffleBuf) free(m_rgShuffleBuf); }
#endif

  bool Init(
    int             a_argc,
    SOCHAR         *a_argv[],
    const SOption *a_rgOptions,
    int             a_nFlags = 0
  );

  inline void SetOptions(const SOption *a_rgOptions) {
    m_rgOptions = a_rgOptions;
  }

  inline void SetFlags(int a_nFlags) { m_nFlags = a_nFlags; }

  inline bool HasFlag(int a_nFlag) const {
    return (m_nFlags & a_nFlag) == a_nFlag;
  }

  bool Next();

  void Stop();

  inline ESOError LastError() const  { return m_nLastError; }

  inline int OptionId() const { return m_nOptionId; }

  inline const SOCHAR *OptionText() const { return m_pszOptionText; }

  inline SOCHAR *OptionArg() const { return m_pszOptionArg; }

  SOCHAR **MultiArg(int n);

  inline int FileCount() const { return m_argc - m_nLastArg; }

  inline SOCHAR *File(int n) const {
    SO_ASSERT(n >= 0 && n < FileCount());
    return m_argv[m_nLastArg + n];
  }

  inline SOCHAR **Files() const { return &m_argv[m_nLastArg]; }

 private:
  CSimpleOptTempl(const CSimpleOptTempl &); // disabled
  CSimpleOptTempl &operator=(const CSimpleOptTempl &);  // disabled

  SOCHAR PrepareArg(SOCHAR *a_pszString) const;
  bool NextClumped();
  void ShuffleArg(int a_nStartIdx, int a_nCount);
  int LookupOption(const SOCHAR *a_pszOption) const;
  int CalcMatch(const SOCHAR *a_pszSource, const SOCHAR *a_pszTest) const;

  // Find the '=' character within a string.
  inline SOCHAR *FindEquals(SOCHAR *s) const {
    while (*s && *s != (SOCHAR)'=') ++s;
    return *s ? s : NULL;
  }
  bool IsEqual(SOCHAR a_cLeft, SOCHAR a_cRight, int a_nArgType) const;

  inline void Copy(SOCHAR **ppDst, SOCHAR **ppSrc, int nCount) const {
#ifdef SO_MAX_ARGS
    // keep our promise of no CLIB usage
    while (nCount-- > 0) *ppDst++ = *ppSrc++;
#else
    memcpy(ppDst, ppSrc, nCount * sizeof(SOCHAR *));
#endif
  }

 private:
  const SOption *m_rgOptions;
  int             m_nFlags;
  int             m_nOptionIdx;
  int             m_nOptionId;
  int             m_nNextOption;
  int             m_nLastArg;
  int             m_argc;
  SOCHAR        **m_argv;
  const SOCHAR   *m_pszOptionText;
  SOCHAR         *m_pszOptionArg;
  SOCHAR         *m_pszClump;
  SOCHAR          m_szShort[3];
  ESOError        m_nLastError;
  SOCHAR        **m_rgShuffleBuf;
};

// ---------------------------------------------------------------------------
//                                  IMPLEMENTATION
// ---------------------------------------------------------------------------

template<class SOCHAR>
bool
CSimpleOptTempl<SOCHAR>::Init(
  int             a_argc,
  SOCHAR         *a_argv[],
  const SOption *a_rgOptions,
  int             a_nFlags
) {
  m_argc           = a_argc;
  m_nLastArg       = a_argc;
  m_argv           = a_argv;
  m_rgOptions      = a_rgOptions;
  m_nLastError     = SO_SUCCESS;
  m_nOptionIdx     = 0;
  m_nOptionId      = -1;
  m_pszOptionText  = NULL;
  m_pszOptionArg   = NULL;
  m_nNextOption    = (a_nFlags & SO_O_USEALL) ? 0 : 1;
  m_szShort[0]     = (SOCHAR)'-';
  m_szShort[2]     = (SOCHAR)'\0';
  m_nFlags         = a_nFlags;
  m_pszClump       = NULL;

#ifdef SO_MAX_ARGS
  if (m_argc > SO_MAX_ARGS) {
    m_nLastError = SO_ARG_INVALID_DATA;
    m_nLastArg = 0;
    return false;
  }
#else
  if (m_rgShuffleBuf) {
    free(m_rgShuffleBuf);
  }
  if (m_argc > SO_STATICBUF) {
    m_rgShuffleBuf = (SOCHAR **) malloc(sizeof(SOCHAR *) * m_argc);
    if (!m_rgShuffleBuf) {
      return false;
    }
  }
#endif

  return true;
}

template<class SOCHAR>
bool
CSimpleOptTempl<SOCHAR>::Next() {
#ifdef SO_MAX_ARGS
  if (m_argc > SO_MAX_ARGS) {
    SO_ASSERT(!"Too many args! Check the return value of Init()!");
    return false;
  }
#endif

  // process a clumped option string if appropriate
  if (m_pszClump && *m_pszClump) {
    // silently discard invalid clumped option
    bool bIsValid = NextClumped();
    while (*m_pszClump && !bIsValid && HasFlag(SO_O_NOERR)) {
      bIsValid = NextClumped();
    }

    // return this option if valid or we are returning errors
    if (bIsValid || !HasFlag(SO_O_NOERR)) {
      return true;
    }
  }
  SO_ASSERT(!m_pszClump || !*m_pszClump);
  m_pszClump = NULL;

  // init for the next option
  m_nOptionIdx    = m_nNextOption;
  m_nOptionId     = -1;
  m_pszOptionText = NULL;
  m_pszOptionArg  = NULL;
  m_nLastError    = SO_SUCCESS;

  // find the next option
  SOCHAR cFirst;
  int nTableIdx = -1;
  int nOptIdx = m_nOptionIdx;
  while (nTableIdx < 0 && nOptIdx < m_nLastArg) {
    SOCHAR *pszArg = m_argv[nOptIdx];
    m_pszOptionArg  = NULL;

    // find this option in the options table
    cFirst = PrepareArg(pszArg);
    if (pszArg[0] == (SOCHAR)'-') {
      // find any combined argument string and remove equals sign
      m_pszOptionArg = FindEquals(pszArg);
      if (m_pszOptionArg) {
        *m_pszOptionArg++ = (SOCHAR)'\0';
      }
    }
    nTableIdx = LookupOption(pszArg);

    // if we didn't find this option but if it is a short form
    // option then we try the alternative forms
    if (nTableIdx < 0
        && !m_pszOptionArg
        && pszArg[0] == (SOCHAR)'-'
        && pszArg[1]
        && pszArg[1] != (SOCHAR)'-'
        && pszArg[2]) {
      // test for a short-form with argument if appropriate
      if (HasFlag(SO_O_SHORTARG)) {
        m_szShort[1] = pszArg[1];
        int nIdx = LookupOption(m_szShort);
        if (nIdx >= 0
            && (m_rgOptions[nIdx].nArgType == SO_REQ_CMB
                || m_rgOptions[nIdx].nArgType == SO_OPT)) {
          m_pszOptionArg = &pszArg[2];
          pszArg         = m_szShort;
          nTableIdx      = nIdx;
        }
      }

      // test for a clumped short-form option string and we didn't
      // match on the short-form argument above
      if (nTableIdx < 0 && HasFlag(SO_O_CLUMP))  {
        m_pszClump = &pszArg[1];
        ++m_nNextOption;
        if (nOptIdx > m_nOptionIdx) {
          ShuffleArg(m_nOptionIdx, nOptIdx - m_nOptionIdx);
        }
        return Next();
      }
    }

    // The option wasn't found. If it starts with a switch character
    // and we are not suppressing errors for invalid options then it
    // is reported as an error, otherwise it is data.
    if (nTableIdx < 0) {
      if (!HasFlag(SO_O_NOERR) && pszArg[0] == (SOCHAR)'-') {
        m_pszOptionText = pszArg;
        break;
      }

      pszArg[0] = cFirst;
      ++nOptIdx;
      if (m_pszOptionArg) {
        *(--m_pszOptionArg) = (SOCHAR)'=';
      }
    }
  }

  // end of options
  if (nOptIdx >= m_nLastArg) {
    if (nOptIdx > m_nOptionIdx) {
      ShuffleArg(m_nOptionIdx, nOptIdx - m_nOptionIdx);
    }
    return false;
  }
  ++m_nNextOption;

  // get the option id
  ESOArgType nArgType = SO_NONE;
  if (nTableIdx < 0) {
    m_nLastError    = (ESOError) nTableIdx; // error code
  } else {
    m_nOptionId     = m_rgOptions[nTableIdx].nId;
    m_pszOptionText = m_rgOptions[nTableIdx].pszArg;

    // ensure that the arg type is valid
    nArgType = m_rgOptions[nTableIdx].nArgType;
    switch (nArgType) {
    case SO_NONE:
      if (m_pszOptionArg) {
        m_nLastError = SO_ARG_INVALID;
      }
      break;

    case SO_REQ_SEP:
      if (m_pszOptionArg) {
        // they wanted separate args, but we got a combined one,
        // unless we are pedantic, just accept it.
        if (HasFlag(SO_O_PEDANTIC)) {
          m_nLastError = SO_ARG_INVALID_TYPE;
        }
      }
      // more processing after we shuffle
      break;

    case SO_REQ_CMB:
      if (!m_pszOptionArg) {
        m_nLastError = SO_ARG_MISSING;
      }
      break;

    case SO_OPT:
      // nothing to do
      break;

    case SO_MULTI:
      // nothing to do. Caller must now check for valid arguments
      // using GetMultiArg()
      break;
    }
  }

  // shuffle the files out of the way
  if (nOptIdx > m_nOptionIdx) {
    ShuffleArg(m_nOptionIdx, nOptIdx - m_nOptionIdx);
  }

  // we need to return the separate arg if required, just re-use the
  // multi-arg code because it all does the same thing
  if (   nArgType == SO_REQ_SEP
         && !m_pszOptionArg
         && m_nLastError == SO_SUCCESS) {
    SOCHAR **ppArgs = MultiArg(1);
    if (ppArgs) {
      m_pszOptionArg = *ppArgs;
    }
  }

  return true;
}

template<class SOCHAR>
void
CSimpleOptTempl<SOCHAR>::Stop() {
  if (m_nNextOption < m_nLastArg) {
    ShuffleArg(m_nNextOption, m_nLastArg - m_nNextOption);
  }
}

template<class SOCHAR>
SOCHAR
CSimpleOptTempl<SOCHAR>::PrepareArg(
  SOCHAR *a_pszString
) const {
#ifdef _WIN32
  // On Windows we can accept the forward slash as a single character
  // option delimiter, but it cannot replace the '-' option used to
  // denote stdin. On Un*x paths may start with slash so it may not
  // be used to start an option.
  if (!HasFlag(SO_O_NOSLASH)
      && a_pszString[0] == (SOCHAR)'/'
      && a_pszString[1]
      && a_pszString[1] != (SOCHAR)'-') {
    a_pszString[0] = (SOCHAR)'-';
    return (SOCHAR)'/';
  }
#endif
  return a_pszString[0];
}

template<class SOCHAR>
bool
CSimpleOptTempl<SOCHAR>::NextClumped() {
  // prepare for the next clumped option
  m_szShort[1]    = *m_pszClump++;
  m_nOptionId     = -1;
  m_pszOptionText = NULL;
  m_pszOptionArg  = NULL;
  m_nLastError    = SO_SUCCESS;

  // lookup this option, ensure that we are using exact matching
  int nSavedFlags = m_nFlags;
  m_nFlags = SO_O_EXACT;
  int nTableIdx = LookupOption(m_szShort);
  m_nFlags = nSavedFlags;

  // unknown option
  if (nTableIdx < 0) {
    m_pszOptionText = m_szShort; // invalid option
    m_nLastError = (ESOError) nTableIdx; // error code
    return false;
  }

  // valid option
  m_pszOptionText = m_rgOptions[nTableIdx].pszArg;
  ESOArgType nArgType = m_rgOptions[nTableIdx].nArgType;
  if (nArgType == SO_NONE) {
    m_nOptionId = m_rgOptions[nTableIdx].nId;
    return true;
  }

  if (nArgType == SO_REQ_CMB && *m_pszClump) {
    m_nOptionId = m_rgOptions[nTableIdx].nId;
    m_pszOptionArg = m_pszClump;
    while (*m_pszClump) ++m_pszClump; // must point to an empty string
    return true;
  }

  // invalid option as it requires an argument
  m_nLastError = SO_ARG_MISSING;
  return true;
}

// Shuffle arguments to the end of the argv array.
//
// For example:
//      argv[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8" };
//
//  ShuffleArg(1, 1) = { "0", "2", "3", "4", "5", "6", "7", "8", "1" };
//  ShuffleArg(5, 2) = { "0", "1", "2", "3", "4", "7", "8", "5", "6" };
//  ShuffleArg(2, 4) = { "0", "1", "6", "7", "8", "2", "3", "4", "5" };
template<class SOCHAR>
void
CSimpleOptTempl<SOCHAR>::ShuffleArg(
  int a_nStartIdx,
  int a_nCount
) {
  SOCHAR *staticBuf[SO_STATICBUF];
  SOCHAR **buf = m_rgShuffleBuf ? m_rgShuffleBuf : staticBuf;
  int nTail = m_argc - a_nStartIdx - a_nCount;

  // make a copy of the elements to be moved
  Copy(buf, m_argv + a_nStartIdx, a_nCount);

  // move the tail down
  Copy(m_argv + a_nStartIdx, m_argv + a_nStartIdx + a_nCount, nTail);

  // append the moved elements to the tail
  Copy(m_argv + a_nStartIdx + nTail, buf, a_nCount);

  // update the index of the last unshuffled arg
  m_nLastArg -= a_nCount;
}

// match on the long format strings. partial matches will be
// accepted only if that feature is enabled.
template<class SOCHAR>
int
CSimpleOptTempl<SOCHAR>::LookupOption(
  const SOCHAR *a_pszOption
) const {
  int nBestMatch = -1;    // index of best match so far
  int nBestMatchLen = 0;  // matching characters of best match
  int nLastMatchLen = 0;  // matching characters of last best match

  for (int n = 0; m_rgOptions[n].nId >= 0; ++n) {
    // the option table must use hyphens as the option character,
    // the slash character is converted to a hyphen for testing.
    SO_ASSERT(m_rgOptions[n].pszArg[0] != (SOCHAR)'/');

    int nMatchLen = CalcMatch(m_rgOptions[n].pszArg, a_pszOption);
    if (nMatchLen == -1) {
      return n;
    }
    if (nMatchLen > 0 && nMatchLen >= nBestMatchLen) {
      nLastMatchLen = nBestMatchLen;
      nBestMatchLen = nMatchLen;
      nBestMatch = n;
    }
  }

  // only partial matches or no match gets to here, ensure that we
  // don't return a partial match unless it is a clear winner
  if (HasFlag(SO_O_EXACT) || nBestMatch == -1) {
    return SO_OPT_INVALID;
  }
  return (nBestMatchLen > nLastMatchLen) ? nBestMatch : SO_OPT_MULTIPLE;
}

// calculate the number of characters that match (case-sensitive)
// 0 = no match, > 0 == number of characters, -1 == perfect match
template<class SOCHAR>
int
CSimpleOptTempl<SOCHAR>::CalcMatch(
  const SOCHAR   *a_pszSource,
  const SOCHAR   *a_pszTest
) const {
  if (!a_pszSource || !a_pszTest) {
    return 0;
  }

  // determine the argument type
  int nArgType = SO_O_ICASE_LONG;
  if (a_pszSource[0] != '-') {
    nArgType = SO_O_ICASE_WORD;
  } else if (a_pszSource[1] != '-' && !a_pszSource[2]) {
    nArgType = SO_O_ICASE_SHORT;
  }

  // match and skip leading hyphens
  while (*a_pszSource == (SOCHAR)'-' && *a_pszSource == *a_pszTest) {
    ++a_pszSource;
    ++a_pszTest;
  }
  if (*a_pszSource == (SOCHAR)'-' || *a_pszTest == (SOCHAR)'-') {
    return 0;
  }

  // find matching number of characters in the strings
  int nLen = 0;
  while (*a_pszSource && IsEqual(*a_pszSource, *a_pszTest, nArgType)) {
    ++a_pszSource;
    ++a_pszTest;
    ++nLen;
  }

  // if we have exhausted the source...
  if (!*a_pszSource) {
    // and the test strings, then it's a perfect match
    if (!*a_pszTest) {
      return -1;
    }

    // otherwise the match failed as the test is longer than
    // the source. i.e. "--mant" will not match the option "--man".
    return 0;
  }

  // if we haven't exhausted the test string then it is not a match
  // i.e. "--mantle" will not best-fit match to "--mandate" at all.
  if (*a_pszTest) {
    return 0;
  }

  // partial match to the current length of the test string
  return nLen;
}

template<class SOCHAR>
bool
CSimpleOptTempl<SOCHAR>::IsEqual(
  SOCHAR  a_cLeft,
  SOCHAR  a_cRight,
  int     a_nArgType
) const {
  // if this matches then we are doing case-insensitive matching
  if (m_nFlags & a_nArgType) {
    if (a_cLeft  >= 'A' && a_cLeft  <= 'Z') a_cLeft  += 'a' - 'A';
    if (a_cRight >= 'A' && a_cRight <= 'Z') a_cRight += 'a' - 'A';
  }
  return a_cLeft == a_cRight;
}

// calculate the number of characters that match (case-sensitive)
// 0 = no match, > 0 == number of characters, -1 == perfect match
template<class SOCHAR>
SOCHAR **
CSimpleOptTempl<SOCHAR>::MultiArg(
  int a_nCount
) {
  // ensure we have enough arguments
  if (m_nNextOption + a_nCount > m_nLastArg) {
    m_nLastError = SO_ARG_MISSING;
    return NULL;
  }

  // our argument array
  SOCHAR **rgpszArg = &m_argv[m_nNextOption];

  // Ensure that each of the following don't start with an switch character.
  // Only make this check if we are returning errors for unknown arguments.
  if (!HasFlag(SO_O_NOERR)) {
    for (int n = 0; n < a_nCount; ++n) {
      SOCHAR ch = PrepareArg(rgpszArg[n]);
      if (rgpszArg[n][0] == (SOCHAR)'-') {
        rgpszArg[n][0] = ch;
        m_nLastError = SO_ARG_INVALID_DATA;
        return NULL;
      }
      rgpszArg[n][0] = ch;
    }
  }

  // all good
  m_nNextOption += a_nCount;
  return rgpszArg;
}


// ---------------------------------------------------------------------------
//                                  TYPE DEFINITIONS
// ---------------------------------------------------------------------------

typedef CSimpleOptTempl<char>    CSimpleOptA;

typedef CSimpleOptTempl<wchar_t> CSimpleOptW;

#if defined(_UNICODE)

# define CSimpleOpt CSimpleOptW
#else

# define CSimpleOpt CSimpleOptA
#endif

#endif // INCLUDED_SimpleOpt

