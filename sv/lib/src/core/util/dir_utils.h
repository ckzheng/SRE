#ifndef __DIR_UTILE_H__
#define __DIR_UTILE_H__
#include <string>
#include <vector>
#include "base/idec_types.h"

namespace idec {
#define  MAXFNAMELEN 1034

// operation of directory
class Directory {
 public:
  static bool Exists(const char *path_name);
  static bool Create(const char *path_name);
  static bool IsWritable(const char *path_name);
  static bool Delete(std::string path, bool recursive, char *error_str);
};

// operation of files
class File {
 public:
  static bool  IsReadable(const char *file_name);
  static bool  IsWritable(const char *file_name);
  static bool  IsExistence(const char *file_name);
  static int64 FileLength(const char *file_name);
  static bool  ReadAllBytes(const char *file_name, char *&buf, int64 &buf_size);
  static bool  ReadAllLines(const char *file_name,
                            std::vector<std::string> *all_lines);
  static bool  Delete(const char *file_name);
};


// operation of path strings
class Path {
 public:
  static  char        *Combine(const char *top_path, const char *old_name,
                               char *new_name);
  static  std::string  Combine(std::string path1, std::string path2);
  static  std::string  GetFileName(const std::string &file_path);
  static  char        *GetFileName(const char *file_path);
  static  std::string  GetDirectoryName(const std::string &file_path);
  static  char        *GetDirectoryName(const char *file_path, char *path);


  static bool IsRelativePath(const char *file_name);
  static bool IsAbsolutePath(const char *file_name);

  // normalize the slash as the system default
  static char *Normalize(char *file_name, bool keep_final_slash);
  static char *Normalize(char *file_name);
  static std::string Normalize(std::string &file_name, bool keep_final_slash);
  static char *Normalize(const char *old_name, char *new_name,
                         bool keep_final_slash);
};
};
#endif
