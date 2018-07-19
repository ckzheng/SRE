#ifndef TIME_H
#define TIME_H

#include <string>
namespace idec {
class TimeUtils {

 public:

  // return the current time measured in milliseconds
  static double GetTimeMilliseconds();

  // return the date and time in a string
  static void GetDateTime(std::string &strDateTime);

  // convert hundreths of a second
  static void ConvertHundredths(double dHundredths, int &iHours, int &iMinutes,
                                int &iSeconds) {

    return ConvertMilliseconds(dHundredths*10,iHours,iMinutes,iSeconds);
  }

  // convert milliseconds
  static void ConvertMilliseconds(double dMilliseconds, int &iHours,
                                  int &iMinutes, int &iSeconds) {

    iHours = (int)(dMilliseconds/(1000*3600));
    iMinutes = (int)(dMilliseconds/(1000*60)-iHours*60);
    iSeconds = (int)(dMilliseconds/(1000)-iHours*3600-iMinutes*60);
  }
};

};    // end-of-namespace

#endif
