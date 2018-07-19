#include "util/file_input.h"
#include "base/log_message.h"


namespace idec {
using namespace std;

// constructor
FileInput::FileInput(const char *strFile, bool bBinary) {
  file_name_ = strFile;
  binary_mode_ = bBinary;
}

// destructor
FileInput::~FileInput() {

}

// open the file
IDEC_RETCODE FileInput::Open() {
  if (binary_mode_) {
    iss_.open(file_name_.c_str(), ios::binary);
  } else {
    iss_.open(file_name_.c_str());
  }
  if (!iss_.is_open()) {
    IDEC_WARNING << "unable to open the input file: \"" << file_name_.c_str() <<
                 "\"";
    return IDEC_OPEN_ERROR;
  }
  return IDEC_SUCCESS;
}

void FileInput::OpenOrFail() {
  if (binary_mode_) {
    iss_.open(file_name_.c_str(), ios::binary);
  } else {
    iss_.open(file_name_.c_str());
  }
  if (!iss_.is_open()) {
    IDEC_ERROR << "unable to open the input file: \"" << file_name_.c_str() <<
               "\"";
  }
}

// open the file
IDEC_RETCODE FileInput::Close() {

  iss_.close();
  return IDEC_SUCCESS;
}

// return the file size in bytes
long FileInput::size() {

  iss_.seekg(0, ios::end);
  std::streamoff num_bytes = iss_.tellg();
  iss_.seekg(0, ios::beg);

  return (long)num_bytes;
}

};    // end-of-namespace



