#ifndef FILEINPUT_H
#define FILEINPUT_H

#include <fstream>
#include <iostream>
#include "base/idec_return_code.h"


namespace idec {

class FileInput {

 public:

  // constructor
  FileInput(const char *file_name, bool is_binary);

  // destructor
  ~FileInput();

  // open the file
  IDEC_RETCODE Open();
  void OpenOrFail();

  // close the file
  IDEC_RETCODE Close();

  // return the file size in bytes
  long size();

  // return the stream
  std::istream &GetStream() {

    return iss_;
  }


 private:

  bool binary_mode_;            // whether binary or text mode
  std::ifstream iss_;
  std::string file_name_;

};

};    // end-of-namespace

#endif
