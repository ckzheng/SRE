#ifndef FILEOUTPUT_H
#define FILEOUTPUT_H

#include <fstream>
#include <iostream>
#include <string>
#include "base/idec_return_code.h"
#include "base/idec_types.h"


namespace idec {
class FileOutput {


 public:

  // constructor
  FileOutput(const char *file_name, bool bBinary);

  // destructor
  ~FileOutput();

  // open the file
  IDEC_RETCODE Open();

  // close the file
  void Close();

  // return the stream
  std::ostream &GetStream() {

    return oss_;
  }

  static void SafeWrite(std::ostream &os, char *base, uint64 size);

 private:

  bool binary_mode_;            // whether binary or text mode
  std::ofstream oss_;
  std::string file_name_;

};

};    // end-of-namespace

#endif
