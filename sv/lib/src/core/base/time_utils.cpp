#include <string.h>
#include "base/time_utils.h"


#if defined __linux__ || defined __APPLE__ || __MINGW32__
#include <sys/time.h>
#elif _MSC_VER
#define NOMINMAX
#include <windows.h>
#endif

namespace idec {
using namespace std;
// return the current time measured in milliseconds
double TimeUtils::GetTimeMilliseconds() {

#if defined __linux__ || defined __APPLE__ || __MINGW32__
  struct timeval tv;
  struct timezone tz;
  struct tm *tm = NULL;
  gettimeofday(&tv, &tz);
  tm=localtime(&tv.tv_sec);
  return (tv.tv_sec*1000 + tv.tv_usec/1000);
#elif _MSC_VER
  LARGE_INTEGER time_cur;
  LARGE_INTEGER freq;
  QueryPerformanceCounter(&time_cur);
  if (QueryPerformanceFrequency(&freq) == 0) return
      0.0;  // Hardware does not support this.
  return ((double)time_cur.QuadPart *1000.0) /
         ((double)freq.QuadPart);
#endif

}

// return the date and time in a string
void TimeUtils::GetDateTime(string &strDateTime) {

#if defined __linux__ || defined __APPLE__ || __MINGW32__
  char strAux[100];

  time_t timeAux;
  struct tm *tmTime;

  time(&timeAux);
  tmTime = localtime (&timeAux);

  // get the date and time in string format
  asctime_r(tmTime,strAux);
  // remove the end of line
  while(strAux[strlen(strAux)-1] == '\n') {
    strAux[strlen(strAux)-1] = 0;
  }

  strDateTime = strAux;
#elif _MSC_VER
#pragma warning(disable:4996)
  char strAux[100];
  SYSTEMTIME t;
  //GetSystemTime(&t);    //(this is UTC time)
  GetLocalTime(&t);
  sprintf(strAux, "%04d/%02d/%02d %02d:%02d:%02d:%04d", t.wYear, t.wMonth,
          t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
  strDateTime = strAux;
#endif
}

};    // end-of-namespace



