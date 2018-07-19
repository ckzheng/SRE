#include "util/io_base.h"
#include <vector>
#include <string>
#include "base/log_message.h"

namespace idec {
// read bool value
void IOBase::Read(std::istream &is, bool *data, bool bBinary) {
  if (bBinary) {
    is.read(reinterpret_cast<char *>(data), sizeof(bool));
  } else {
    is >> *data;
  }
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read unsigned char value
void IOBase::Read(std::istream &is, char *iData, bool bBinary) {
  if (bBinary) {
    is.read(iData, sizeof(char));
  } else {
    is >> *iData;
  }
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read unsigned char value
void IOBase::Read(std::istream &is, unsigned char *iData, bool bBinary) {
  if (bBinary) {
    is.read(reinterpret_cast<char *>(iData), sizeof(unsigned char));
  } else {
    is >> *iData;
  }
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read int value
void IOBase::Read(std::istream &is, int *iData, bool bBinary) {
  if (bBinary) {
    is.read(reinterpret_cast<char *>(iData), sizeof(int));
  } else {
    is >> *iData;
  }
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read unsigned int value
void IOBase::Read(std::istream &is, unsigned int *iData, bool bBinary) {
  if (bBinary) {
    is.read(reinterpret_cast<char *>(iData), sizeof(unsigned int));
  } else {
    is >> *iData;
  }
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read float value
void IOBase::Read(std::istream &is, float *fData, bool bBinary) {
  if (bBinary) {
    is.read(reinterpret_cast<char *>(fData), sizeof(float));
  } else {
    is >> *fData;
  }
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read double value
void IOBase::Read(std::istream &is, double *dData, bool bBinary) {
  if (bBinary) {
    is.read(reinterpret_cast<char *>(dData), sizeof(double));
  } else {
    is >> *dData;
  }
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read a string
void IOBase::ReadString(std::istream &is, char **str) {
  // read length
  int iElements = -1;
  is.read(reinterpret_cast<char *>(&iElements), sizeof(int));
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }

  // read the actual elements
  *str = new char[iElements + 1];
  is.read(*str, sizeof(char)*iElements);
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
  (*str)[iElements] = 0;
}

// read a string
void IOBase::ReadString(std::istream &is, std::string &str, bool bBinary) {
  if (!bBinary) {
    is >> str;
    if (is.fail()) {
      IDEC_ERROR << "error reading from stream at position: " << is.tellg();
    }
  } else {
    char *p = NULL;
    ReadString(is, &p);
    str = p;
    delete[]p;
  }
}

// read bytes
void IOBase::ReadBytes(std::istream &is, char *cData, int iBytes) {
  is.read(cData, iBytes);
  if (is.fail()) {
    IDEC_ERROR << "error reading from stream at position: " << is.tellg();
  }
}

// read whitespace (text mode)
void IOBase::ReadWhiteSpaces(std::istream &is) {
  char c;
  while (is.peek() == ' ') {
    is >> c;
  }
}

// read all the context into one string
void IOBase::ReadAllText(std::istream &iss, std::string *out_str) {
  iss.seekg(0, std::ios::end);
  out_str->reserve(iss.tellg());
  iss.seekg(0, std::ios::beg);

  out_str->assign((std::istreambuf_iterator<char>(iss)),
                  std::istreambuf_iterator<char>());
}


// write bool value
void IOBase::Write(std::ostream &os, bool bData, bool bBinary) {
  if (bBinary) {
    os.write(reinterpret_cast<char *>(&bData), sizeof(bool));
  } else {
    os << bData;
  }
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

// write unsigned char value
void IOBase::Write(std::ostream &os, unsigned char iData, bool bBinary) {
  if (bBinary) {
    os.write(reinterpret_cast<char *>(&iData), sizeof(unsigned char));
  } else {
    os << iData;
  }
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

// write int value
void IOBase::Write(std::ostream &os, int iData, bool bBinary) {
  if (bBinary) {
    os.write(reinterpret_cast<char *>(&iData), sizeof(int));
  } else {
    os << iData;
  }
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

// write int value
void IOBase::Write(std::ostream &os, unsigned int iData, bool bBinary) {
  if (bBinary) {
    os.write(reinterpret_cast<char *>(&iData), sizeof(unsigned int));
  } else {
    os << iData;
  }
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

// write float value
void IOBase::Write(std::ostream &os, float fData, bool bBinary) {
  if (bBinary) {
    os.write(reinterpret_cast<char *>(&fData), sizeof(float));
  } else {
    os << fData;
  }
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

// write double value
void IOBase::Write(std::ostream &os, double dData, bool bBinary) {
  if (bBinary) {
    os.write(reinterpret_cast<char *>(&dData), sizeof(double));
  } else {
    os << dData;
  }
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

// write a string
void IOBase::WriteCString(std::ostream &os, const char *str, int iLength) {
  // write length
  os.write(reinterpret_cast<char *>(&iLength), sizeof(int));
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }

  // write the actual elements
  os.write(str, sizeof(char)*iLength);
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

// write a string
void IOBase::WriteString(std::ostream &os,
                         const std::string &str,
                         bool bBinary) {
  if (bBinary) {
    WriteCString(os, str.c_str(), static_cast<int>(str.size()));
  } else {
    os << str;
    if (os.fail()) {
      IDEC_ERROR << "error writing to stream";
    }
  }
}

// write a string
void IOBase::WriteString(std::ostream &os, std::ostringstream &oss) {
  std::
  string str = oss.str();
  os << str;
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}



// write double value
void IOBase::WriteBytes(std::ostream &os, char *bData, int iBytes) {
  os.write(bData, iBytes);
  if (os.fail()) {
    IDEC_ERROR << "error writing to stream";
  }
}

};    // namespace idec

