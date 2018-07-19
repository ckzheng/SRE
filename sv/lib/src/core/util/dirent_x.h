#ifndef _IDEC_DIRENT_H_
#define _IDEC_DIRENT_H_

#ifdef WIN32
#pragma warning(disable:4996)
#endif

#ifndef	PATH_DELIMITERA
#ifdef WIN32
#define PATH_DELIMITERA "\\"
#define PATH_DELIMITERA_CHAR '\\'
#else
#define PATH_DELIMITERA "/"
#define PATH_DELIMITERA_CHAR '/'
#endif // WIN32
#endif

#ifdef WIN32
#include <stdio.h>
#include <io.h>
#include <errno.h>
#include <tchar.h>
#include <direct.h>
#include <sys/STAT.H>
#include <stdlib.h>
#include <string.h>
#define FX_MAX_PATH 512
#ifndef S_ISDIR
#define S_ISDIR(m)	(((m)&S_IFDIR)==S_IFDIR)
#endif

#ifndef FILENAME_MAX
#define FILENAME_MAX 260
#endif

namespace idec {


struct dirent {
  long            d_ino;					/* Always zero. */
  unsigned short  d_reclen;				/* Always zero. */
  unsigned short  d_namlen;				/* Length of name in d_name. */
  char           d_name[FILENAME_MAX];	/* File name. */
};

/**
 * This is an internal data structure. Good programmers will not use it
 * except as an argument to one of the functions below.
 * dd_stat field is now int (was short in older versions).
 */
typedef struct DIR {
  /* disk transfer area for this dir */
  struct _finddata_t     dd_dta;

  /* dirent struct to return from dir (NOTE: this makes this thread
  * safe as long as only one thread uses a particular DIR struct at
  * a time) */
  struct dirent           dd_dir;

  /* _findnext handle */
  long                    dd_handle;

  /**
   * Status of search:
   *   0 = not started yet (next entry to read is first entry)
   *  -1 = off the end
   *   positive = 0 based index of next entry
   */
  int                     dd_stat;

  /* given path for dir with search pattern (struct is extended) */
  char                    dd_name[FX_MAX_PATH];
} DIR;


#define DIRENT_SEARCH_SUFFIX "*"
#define DIRENT_SLASH PATH_DELIMITERA

DIR			*opendir(const char *);
struct dirent	*readdir( DIR *);
int				closedir( DIR *);
/*void 			rewinddir(DIR*);
int32_t 		telldir(DIR*);
void 			seekdir(DIR*, long);*/

//FX_NS_END
};
#else //for linux
# include <dirent.h>
#endif


#endif  /* Not _DIRENT_H_ */
