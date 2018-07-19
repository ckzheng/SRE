#ifndef  ASR_DECODER_SRC_CORE_UTIL_IO_BASE_H_
#define  ASR_DECODER_SRC_CORE_UTIL_IO_BASE_H_

#include <vector>
#include <string>
#include "util/file_input.h"
#include "base/log_message.h"
#include "base/idec_types.h"


namespace idec {
class IOBase {
 public:
  // read bool value
  static void Read(std::istream &is, bool *bData, bool bBinary = true);

  // read unsigned char value
  static void Read(std::istream &is, char *iData, bool bBinary = true);

  // read unsigned char value
  static void Read(std::istream &is, unsigned char *iData, bool bBinary = true);

  // read int value
  static void Read(std::istream &is, int *iData, bool bBinary = true);

  // read unsigned int value
  static void Read(std::istream &is, unsigned int *iData, bool bBinary = true);

  // read float value
  static void Read(std::istream &is, float *fData, bool bBinary = true);

  // read double value
  static void Read(std::istream &is, double *fData, bool bBinary = true);

  // read a string
  static void ReadString(std::istream &is, char **str);
  // read a string
  static void ReadString(std::istream &is, std::string &str,
                         bool bBinary = false);

  // read bytes
  static void ReadBytes(std::istream &is, char *bData, int iBytes);

  // read whitespace (text mode)
  static void ReadWhiteSpaces(std::istream &is);

  // read the whole stream into a string
  static void ReadAllText(std::istream &is, std::string *out_str);

  // write bool value
  static void Write(std::ostream &os, bool bData, bool bBinary = true);

  // write unsigned char value
  static void Write(std::ostream &os, unsigned char iData, bool bBinary = true);

  // write int value
  static void Write(std::ostream &os, int iData, bool bBinary = true);

  // write unsigned int value
  static void Write(std::ostream &os, unsigned int iData, bool bBinary = true);

  // write float value
  static void Write(std::ostream &os, float fData, bool bBinary = true);

  // write double value
  static void Write(std::ostream &os, double dData, bool bBinary = true);

  // write a string
  static void WriteCString(std::ostream &os, const char *str, int iLength);

  // write a string
  static void WriteString(std::ostream &os, const std::string &str,
                          bool bBinary = false);

  // write a string
  static void WriteString(std::ostream &os, std::ostringstream &oss);

  // write double value
  static void WriteBytes(std::ostream &os, char *bData, int iBytes);

  template <typename T>
  static void Read(std::istream &is,  std::vector<T> *v) {
    assert(v != NULL);
    uint32 size = 0;
    is.read(reinterpret_cast<char *>(&size),
            sizeof(size));
    if (size > 0) {
      v->resize(size);
      is.read(reinterpret_cast<char *>(&v[0]), sizeof(T) * v->size());
    }
  }

  template <typename T>
  static void Write(std::ostream &os, const std::vector<T> &v) {
    uint32 size = static_cast<uint32>(v.size());
    IOBase::Write(os, size);
    if (size > 0) {
      os.write(reinterpret_cast<const char *>(v[0]), sizeof(T)*size);
    }
  }
};

};  // namespace idec

#endif  // ASR_DECODER_SRC_CORE_UTIL_IO_BASE_H_

