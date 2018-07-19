#include "serialize_helper.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <istream>
#include <limits>
#include <climits>
#include "base/idec_common.h"
#include "base/idec_types.h"

using std::max;
using std::min;

namespace idec {

const size_t SerializeHelper::serializeBlock;

SerializeHelper::SerializeHelper(size_t expectedMemory) : pos(0) {
  buf.reserve(expectedMemory);
  tmp.reserve(1024);
}


SerializeHelper::~SerializeHelper(void) {

}

//=======================================================================
// file handling functions (by zhijie.yzj)
//=======================================================================
IDEC_RETCODE SerializeHelper::WriteFile(const char *fn) {
  FILE *fp;
  if(fopen_s(&fp, fn, "wb")) {
    fprintf(stderr, "SerializeHelper::writeFile: cannot create %s\n", fn);

    // retry
    size_t n = 0;
    errno_t err;
    do {
#ifdef _MSC_VER
      Sleep(5000);
#else
      sleep(5);
#endif
      err = fopen_s(&fp, fn, "wb");
    } while(err != 0 && ++n < 5);

    if(err != 0) {
#ifdef _MSC_VER
      char errMsg[2048];
      strerror_s(errMsg, 2048, err);
#else
      char *errMsg = strerror(err);
#endif
      fprintf(stderr,
              "SerializeHelper::writeFile: cannot create %s after retry, %s\n", fn, errMsg);
    }
    return IDEC_OPEN_ERROR;
  }

  // write a placeholder first
  int32 len = 0;
  fwrite(&len, sizeof(len), 1, fp);

  // write data block by block
  size_t pos = 0;
  while(pos < buf.size()) {
    fwrite(&buf[0]+pos, sizeof(BYTE), min(serializeBlock, buf.size()-pos), fp);
    pos += serializeBlock;
  }

  // write the length to the placeholder, indicating successful writing
#ifdef _MSC_VER
  _fseeki64(fp, 0, SEEK_SET);
#elif defined __BIONIC__ || defined __APPLE__
  fseek(fp, 0, SEEK_SET);
#else
  fseeko64(fp, 0, SEEK_SET);
#endif
  len = (int32)(buf.size() % INT_MAX);
  fwrite(&len, sizeof(len), 1, fp);

  fclose(fp);

  return IDEC_SUCCESS;
}

void SerializeHelper::Write(std::ostream &oss) {
  // write a placeholder first
  int32 len = (int32)(buf.size() % INT_MAX);
  oss.write(reinterpret_cast<const char *>(&len), sizeof(len));

  // write data block by block
  size_t pos = 0;
  while (pos < buf.size()) {
    oss.write(reinterpret_cast<const char *>(&buf[0] + pos),
              sizeof(BYTE) * min(serializeBlock, buf.size() - pos));
    pos += serializeBlock;
  }
}



IDEC_RETCODE SerializeHelper::ReadFile(const char *fn) {
  FILE *fp;
  if(fopen_s(&fp, fn, "rb")) {
    fprintf(stderr, "SerializeHelper::readFile: cannot open %s\n", fn);
    return IDEC_OPEN_ERROR;
  }

  int32 len_expected;
  fread(&len_expected, sizeof(len_expected), 1, fp);
#ifdef _MSC_VER
  _fseeki64(fp, 0, SEEK_END);
  long long len = _ftelli64(fp) - sizeof(int32);
#elif defined __BIONIC__ || defined __APPLE__
  fseek(fp, 0, SEEK_END);
  int32 len = ftell(fp) - sizeof(int32);
#else
  fseeko64(fp, 0, SEEK_END);
  off64_t len = ftello64(fp) - sizeof(int32);
#endif

  if (int32(len % INT_MAX) != len_expected) {
    fprintf(stderr, "SerializeHelper::readFile: bad file %s\n", fn);
    return IDEC_FATAL_ERROR;
  }

#ifdef _MSC_VER
  _fseeki64(fp, sizeof(int32), SEEK_SET);
#elif defined __BIONIC__ || defined __APPLE__
  fseek(fp, sizeof(int32), SEEK_SET);
#else
  fseeko64(fp, sizeof(int32), SEEK_SET);
#endif
  buf.resize(len);

  // read data block by block
  size_t pos = 0;
  while(pos < buf.size()) {
    fread(&buf[0]+pos, sizeof(BYTE), min(serializeBlock, buf.size()-pos), fp);
    pos += serializeBlock;
  }

  fclose(fp);

  pos = 0;
  return IDEC_SUCCESS;
}

void SerializeHelper::Read(std::istream &iss) {
  int32 len_expected;
  iss.read(reinterpret_cast<char *>(&len_expected), sizeof(len_expected));
  buf.resize(len_expected);
  // read data block by block
  size_t pos = 0;
  while (pos < buf.size()) {
    iss.read(reinterpret_cast<char *>(&buf[0] + pos),
             sizeof(BYTE) * min(serializeBlock, buf.size() - pos));
    pos += serializeBlock;
  }
  pos = 0;
}

bool SerializeHelper::CheckFileByLength(const char *fn) {
  FILE *fp;
  if(fopen_s(&fp, fn, "rb")) {
    return(false);
  }

  int32 len_expected;
  if(fread(&len_expected, sizeof(int32), 1, fp) != 1) {
    fclose(fp);
    return(false);
  }
#ifdef _MSC_VER
  _fseeki64(fp, 0, SEEK_END);
  long long len = _ftelli64(fp) - sizeof(int32);
#elif defined __BIONIC__ || defined __APPLE__
  fseek(fp, 0, SEEK_END);
  int len = ftell(fp) - sizeof(int32);
#else
  fseeko64(fp, 0, SEEK_END);
  long long len = ftello64(fp) - sizeof(int32);
#endif
  fclose(fp);

  if (int32(len % INT_MAX) != len_expected) {
    return(false);
  }

  return(true);
}

//=======================================================================
// serialize a memory block given by a pointer (by zhijie.yzj)
//=======================================================================
void SerializeHelper::Serialize(const void *var, size_t bytes) {
  tmp.resize(bytes);
  memcpy_s(&tmp[0], bytes, var, bytes);
  buf.insert(buf.end(), tmp.begin(), tmp.end());
}

void SerializeHelper::Deserialize(void *var, size_t bytes) {
  // CAUTION: var must be correctly allocated
  memcpy_s(var, bytes, &buf[0] + pos, bytes);
  pos += bytes;
}

//=======================================================================
// serialize an STL string (by zhijie.yzj)
//=======================================================================
// void SerializeHelper::Serialize(const std::string &var)
// {
//     Serialize(var.length()+1);
//
//     tmp.resize(var.length()+1);
//     memcpy_s(&tmp[0], var.length()+1, var.c_str(), var.length()+1);
//     buf.insert(buf.end(), tmp.begin(), tmp.end());
// }
//
// void SerializeHelper::Deserialize(std::string &var)
// {
//     size_t len;
//     Deserialize(len);
//
//     var.assign((char*)(&buf[0]+pos));
// #ifdef _DEBUG
//     if(len != var.length()+1)
//     {
//         fprintf(stderr, "SerializeHelper::Deserialize: string format error\n");
//     }
// #endif
//     pos += len;
// }
//
// void SerializeHelper::Serialize(const std::wstring &var)
// {
//     Serialize(var.length()+1);
//
//     tmp.resize((var.length()+1)*2);
//     memcpy_s(&tmp[0], (var.length()+1)*2, var.c_str(), (var.length()+1)*2);
//     buf.insert(buf.end(), tmp.begin(), tmp.end());
// }
//
// void SerializeHelper::Deserialize(std::wstring &var)
// {
//     size_t len;
//     Deserialize(len);
//
//     var.assign((wchar_t*)(&buf[0]+pos));
// #ifdef _DEBUG
//     if(len != var.length()+1)
//     {
//         fprintf(stderr, "SerializeHelper::Deserialize: string format error\n");
//     }
// #endif
//     pos += len*2;
// }

//=======================================================================
// serialize a concurrent vector (by zhijie.yzj)
//=======================================================================
//template <typename T> void SerializeHelper::Serialize(const concurrent_vector<T> &var)
//{
//    serialize(var.size());
//
//    for(size_t i=0; i<var.size(); ++i) serialize(var[i]);
//}
//
//template <typename T> void SerializeHelper::Deserialize(concurrent_vector<T> &var)
//{
//    size_t len;
//    deserialize(len);
//
//    var.resize(len);
//    for(size_t i=0; i<var.size(); ++i) deserialize(var[i]);
//}

//=======================================================================
// serialize an STL unordered_map (by zhijie.yzj)
//=======================================================================

//template <typename T1, typename T2> void SerializeHelper::Serialize(const unordered_map<T1, T2> &var)
//{
//    Serialize(var.size());
//
//    typename unordered_map<T1, T2>::const_iterator pos;
//    for(pos=var.begin(); pos!=var.end(); ++pos)
//    {
//        Serialize(pos->first);
//        Serialize(pos->second);
//    }
//}
//
//template <typename T1, typename T2> void SerializeHelper::Deserialize(std::unordered_map<T1, T2> &var)
//{
//    size_t len;
//    Deserialize(len);
//
//    for(size_t i=0; i<len; ++i)
//    {
//        T1 var1;
//        T2 var2;
//        Deserialize(var1);
//        Deserialize(var2);
//        var.insert(make_pair(var1, var2));
//    }
//}

//=======================================================================
// supported types are listed below (by zhijie.yzj)
//=======================================================================
//template void SerializeHelper::Serialize(const concurrent_vector<double> &var);
//template void SerializeHelper::Deserialize(concurrent_vector<double> &var);
//template void SerializeHelper::Serialize(const concurrent_vector<vector<double>> &var);
//template void SerializeHelper::Deserialize(concurrent_vector<vector<double>> &var);



}
