///////////////////////////////////////////////////////////////////
//
// helper class used for serialization and deserialization
// Copyright 2015 Alibaba-inc  [zhijie.yzj]
//
///////////////////////////////////////////////////////////////////

#ifndef SERIALIZEHELPER_H_
#define SERIALIZEHELPER_H_

#ifdef _MSC_VER
#include <Windows.h>
#else
typedef unsigned char BYTE;
#endif

#include <vector>
#include <string>
#include <memory>
#include <cstdlib>
#include <stdint.h>
#include  <cstring>

#include "base/idec_common.h"
#include "base/idec_return_code.h"

namespace idec {

class SerializeHelper {
 protected:
  std::vector<BYTE> buf;
  std::vector<BYTE> tmp;
  size_t pos;
  static const size_t serializeBlock = 10485760;	// 10MB

 public:
  SerializeHelper(size_t expectedMemory = 10485760);	// 10MB
  ~SerializeHelper(void);

  // memory functions
  void Clear() {buf.clear(); pos=0;}
  size_t GetSize() {return(buf.size());}
  void Resize(size_t len) {buf.resize(len);}
  BYTE *GetPtr() {return(&buf[0]);}
  void Rewind() {pos=0;}
  size_t &GetPos() {return(pos);}

  // file handling functions
  IDEC_RETCODE WriteFile(const char *fn);
  IDEC_RETCODE ReadFile(const char *fn);
  void Write(std::ostream &oss);
  void Read(std::istream &iss);


  // check if file exists and its consistency (check length only, in most cases this is a good indicator of file consistency)
  bool CheckFileByLength(const char *fn);

  // single variable
  template <typename T> void Serialize(const T &var);
  template <typename T> void Deserialize(T &var);

  // pointers
  void Serialize(const void *var, size_t bytes);
  void Deserialize(void *var, size_t bytes);

  // string
// 	void Serialize(const std::string &var);
// 	void Deserialize(std::string &var);
// 	void Serialize(const std::wstring &var);
// 	void Deserialize(std::wstring &var);

  // std vector
  template <typename T> void Serialize(const std::vector<T> &var);
  template <typename T> void Deserialize(std::vector<T> &var);

  // concurrent_vector
  //template <typename T> void Serialize(const concurrent_vector<T> &var);
  //template <typename T> void Deserialize(concurrent_vector<T> &var);

  // std hash_map
#ifndef __INCLUDE_TR1__
  template <typename T1, typename T2> void Serialize(const
      std::unordered_map<T1, T2> &var);
  template <typename T1, typename T2> void Deserialize(std::unordered_map<T1, T2>
      &var);
#else
  template <typename T1, typename T2> void Serialize(const
      std::tr1::unordered_map<T1, T2> &var);
  template <typename T1, typename T2> void Deserialize(
    std::tr1::unordered_map<T1, T2> &var);
#endif

#ifndef _MSC_VER
  // used for linux
 private:
  void memcpy_s(void *_Dst, size_t _DstSize, const void *_Src,
                size_t _MaxCount) {
    memcpy(_Dst, _Src, _MaxCount);
  }
#endif
};


//=======================================================================
// serialize a variable (by zhijie.yzj)
//=======================================================================
template <typename T> void SerializeHelper::Serialize(const T &var) {
  tmp.resize(sizeof(T));
  memcpy_s(&tmp[0], sizeof(T), &var, sizeof(T));
  buf.insert(buf.end(), tmp.begin(), tmp.end());
}

template <typename T> void SerializeHelper::Deserialize(T &var) {
  memcpy_s(&var, sizeof(T), &buf[0] + pos, sizeof(T));
  pos += sizeof(T);
}

//=======================================================================
// serialize an STL vector (by zhijie.yzj)
//=======================================================================
template <typename T> void SerializeHelper::Serialize(const std::vector<T>
    &var) {
  uint32_t count = static_cast<uint32_t>(var.size());
  if (count != var.size()) {
    throw "Unexpected size of a array to be serialized.";
  }

  Serialize(count);

  for (size_t i = 0; i < var.size(); ++i) Serialize(var[i]);
}

template <typename T> void SerializeHelper::Deserialize(std::vector<T> &var) {
  uint32_t len;
  Deserialize(len);

  var.resize(len);
  for (size_t i = 0; i < var.size(); ++i) Deserialize(var[i]);
}

}

#endif
