#include "util/file_output.h"
#include "util/dir_utils.h"
#include "base/log_message.h"
#include <algorithm>



namespace idec {

// constructor
FileOutput::FileOutput(const char *strFile, bool bBinary) {
  file_name_ = strFile;
  binary_mode_ = bBinary;
}

// destructor
FileOutput::~FileOutput() {

}

// open the file
IDEC_RETCODE FileOutput::Open() {

  IDEC_RETCODE ret = IDEC_SUCCESS;
  Directory::Create(file_name_.c_str());
  if (binary_mode_) {
    oss_.open(file_name_.c_str(), std::ios::binary);
  } else {
    oss_.open(file_name_.c_str());
  }
  if (!oss_.is_open()) {
    IDEC_ERROR << "unable to open the file: " << file_name_.c_str();
    ret = IDEC_OPEN_ERROR;
  }
  return ret;
}

// open the file
void FileOutput::Close() {
  oss_.close();
}

void FileOutput::SafeWrite(std::ostream &os, char *base, uint64 size) {
  uint64 max_size = 1024 * 1024 * 1024;
  uint64 write_size = 0;
  uint64 init_p = os.tellp();
  while (write_size < size) {
    uint64 write_this = std::min(max_size, size - write_size);
    os.write(base + write_size, write_this);
    write_size += write_this;
  }
  if ((uint64)os.tellp() - init_p != write_size) {
    IDEC_ERROR << "wrong writing, expected " << write_size << "actual" <<
               (uint64)os.tellp() - init_p<<"\n";
  }
}


};    // end-of-namespace


