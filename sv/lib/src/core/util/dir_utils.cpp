#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#ifndef WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#else
#include <direct.h>
#endif
#include "util/dirent_x.h"
#include "util/dir_utils.h"
#include "base/log_message.h"
namespace idec {
using namespace std;
#ifndef UNIX_DIR_SEPARATOR
#define UNIX_DIR_SEPARATOR '/'
#endif
#ifndef DOS_DIR_SEPARATOR
#define DOS_DIR_SEPARATOR '\\'
#endif
#if defined(_WIN32) || defined(__MSDOS__) || defined(__DJGPP__) || \
defined(__OS2__)
#define HAVE_DOS_BASED_FILE_SYSTEM
// system dependent things
#define  SYS_DEFAULT_DIR_SEPARATOR DOS_DIR_SEPARATOR
#define  ALT_DIR_SEPARATOR UNIX_DIR_SEPARATOR
#else
#define  SYS_DEFAULT_DIR_SEPARATOR UNIX_DIR_SEPARATOR
#define  ALT_DIR_SEPARATOR DOS_DIR_SEPARATOR
#endif
/* Define IS_DIR_SEPARATOR.  */
#ifndef DOS_DIR_SEPARATOR
# define IS_DIR_SEPARATOR(ch) ((ch) == UNIX_DIR_SEPARATOR)
#else /* DOS_DIR_SEPARATOR */
# define IS_DIR_SEPARATOR(ch) \
(((ch) == UNIX_DIR_SEPARATOR) || ((ch) == DOS_DIR_SEPARATOR))
#endif /* DOS_DIR_SEPARATOR */
char *Path::Normalize(char *path) {
  // change the delimiter
  size_t n = strlen(path);
  size_t i = 0;
  for (i = 1; i < n; i++) {
    if (path[i] == ALT_DIR_SEPARATOR) {
      path[i] = SYS_DEFAULT_DIR_SEPARATOR;
    }
  }
  // trim right space
  while (strlen(path) >= 1 && isspace(path[strlen(path) - 1])) {
    path[strlen(path) - 1] = '\0';
  }
  // trim the left space
  while (strlen(path) >= 1 && isspace(path[0])) {
    strcpy(path, path + 1);
  }
  return path;
}
char *Path::Normalize(char *path, bool keep_final_slash) {
  path = Normalize(path);
  if (!keep_final_slash) {
    if (strlen(path) >= 1 &&
        (path[strlen(path) - 1] == SYS_DEFAULT_DIR_SEPARATOR)) {
      path[strlen(path) - 1] = '\0';
    }
  } else {
    if (strlen(path) >= 1 &&
        (path[strlen(path) - 1] != SYS_DEFAULT_DIR_SEPARATOR)) {
      path[strlen(path) + 1] = '\0';
      path[strlen(path)] = SYS_DEFAULT_DIR_SEPARATOR;
    }
  }
  return path;
}
char *Path::GetDirectoryName(const char *file_path, char *dir_name) {
  char *t;
  strcpy(dir_name, file_path);
  Normalize(dir_name);
  t = strrchr(dir_name, SYS_DEFAULT_DIR_SEPARATOR);
  if (t == NULL) {
    *dir_name = '\0';
  } else {
    *++t = '\0';
  }
  return NULL;
}
string Path::GetDirectoryName(const string &file_path) {
  char file_path_cpy[MAXFNAMELEN], dir_path[MAXFNAMELEN];
  strcpy(file_path_cpy, file_path.c_str());
  GetDirectoryName(file_path_cpy, dir_path);
  return string(dir_path);
}
char *Path::GetFileName(const char *name) {
  const char *base;
#if defined (HAVE_DOS_BASED_FILE_SYSTEM)
  // Skip over the disk name in MSDOS pathnames.
  if (isascii(name[0]) && isalpha(name[0]) && name[1] == ':')
    name += 2;
#endif
  for (base = name; *name; name++) {
    if (IS_DIR_SEPARATOR(*name)) {
      base = name + 1;
    }
  }
  return (char *)base;
}

// find the base name of the file path
string Path::GetFileName(const string &file_path) {
  return string(GetFileName(file_path.c_str()));
}

char *Path::Combine(const char *path1, const char *path2, char *ful) {
  if (path1[strlen(path1) - 1] == SYS_DEFAULT_DIR_SEPARATOR
      || path1[strlen(path1) - 1] == ALT_DIR_SEPARATOR) {
    sprintf(ful, "%s%s", path1, path2);
  } else {
    sprintf(ful, "%s%c%s", path1, SYS_DEFAULT_DIR_SEPARATOR, path2);
  }
  Normalize(ful);
  return ful;
}

string Path::Combine(string path1, string path2) {
  string full_path = "";
  Normalize(path1, false);
  Normalize(path2, false);
  // path2 is current path .\ or ./ remove the prefix
  if (path2.size() > 1 && path2[0] == '.' &&
      (path2[1] == SYS_DEFAULT_DIR_SEPARATOR) && !path1.empty()) {
    path2 = path2.substr(2);
  }
  if (path1.empty() || path1[path1.size() - 1] == SYS_DEFAULT_DIR_SEPARATOR) {
    full_path = path1 + path2;
  } else {
    full_path = path1 + SYS_DEFAULT_DIR_SEPARATOR + path2;
  }
  return full_path;
}

bool Path::IsRelativePath(const char *name) {
#if defined (HAVE_DOS_BASED_FILE_SYSTEM)
  // Skip over the disk name in MSDOS pathnames.
  if (isascii(name[0]) && isalpha(name[0]) && name[1] == ':') {
    return false;
  } else {
    return true;
  }
#else
  if (IS_DIR_SEPARATOR(name[0])) {
    return false;
  } else {
    return true;
  }
#endif
}

bool Path::IsAbsolutePath(const char *file_name) {
  return !IsRelativePath(file_name);
}

string Path::Normalize(string &file_name, bool keep_final_slash) {
  char buf[MAXFNAMELEN];
  strcpy(buf, file_name.c_str());
  file_name = Normalize(buf, keep_final_slash);
  return file_name;
}

char *Path::Normalize(const char *old_name, char *new_name,
                      bool keep_final_slash) {
  if (NULL == new_name) {
    return NULL;
  }
  new_name[0] = '\0';
  if (NULL == old_name) {
    return NULL;
  }
  strcpy(new_name, old_name);
  return Normalize(new_name, keep_final_slash);
}

bool Directory::IsWritable(const char *path_name) {
  return (access(path_name, 02) != -1);
}

bool Directory::Exists(const char  *path_name) {
  if (NULL == path_name) return false;
  return (access(path_name, 04) != -1);
}

bool Directory::Create(const char  *file_name) {
  size_t i, n;
  char *pdest;
  char work_name[MAXFNAMELEN];
  n = strlen(file_name);
  strcpy(work_name, file_name);
  Path::Normalize(work_name);
  pdest = strrchr(work_name, SYS_DEFAULT_DIR_SEPARATOR);
  if (pdest != NULL) {
    pdest[0] = '\0';
    n = strlen(work_name);
    for (i = 1; i < n; i++)
      if (work_name[i] == SYS_DEFAULT_DIR_SEPARATOR &&
          ((work_name[i - 1] != '.')
#ifdef WIN32
           || (work_name[i - 1] != ':')
#endif
          )) {
        work_name[i] = '\0';
        if (strlen(work_name)) {
#ifdef WIN32
          _mkdir(work_name);
#else
          mkdir(work_name, 0777);
#endif
          work_name[i] = SYS_DEFAULT_DIR_SEPARATOR;
        }
      }
#ifdef WIN32
    _mkdir(work_name);
#else
    mkdir(work_name, 0777);
#endif
  }
  return true;
}


// 01 X_OK; 02 W_OK; 04 R_OK 0 F_OK;
bool File::IsReadable(const char *file_name) {
  if (NULL == file_name) return false;
  return (access(file_name, 04) != -1);
}

bool File::IsWritable(const char *file_name) {
  if (NULL == file_name) return false;
  return (access(file_name, 02) != -1);
}

bool File::IsExistence(const char *file_name) {
  if (NULL == file_name) return false;
  return (access(file_name, 00) != -1);
}

bool File::Delete(const char *file_name) {
  return (0 == remove(file_name));
}

int64 File::FileLength(const char *file_name) {
  FILE *fp = fopen(file_name, "rb");
  fseek(fp, 0, SEEK_END);
  int64 len = 0;
#ifdef _MSC_VER
  len = _ftelli64(fp);
#elif defined __BIONIC__ || defined __APPLE__
  // android don't support ftello64
  len = ftello(fp);
#else
  len = ftello64(fp);
#endif
  fclose(fp);
  return len;
}

bool File::ReadAllBytes(const char *file_name, char *&buf, int64 &buf_size) {
  FILE *fp = fopen(file_name, "rb");
  if (NULL == fp) {
    return false;
  }
  fseek(fp, 0, SEEK_END);
#ifdef _MSC_VER
  buf_size = _ftelli64(fp);
#elif defined __BIONIC__ || defined __APPLE__
  // android don't support ftello64
  buf_size = ftello(fp);
#else
  buf_size = ftello64(fp);
#endif
  fseek(fp, 0, SEEK_SET);
  buf = new char[buf_size];
  fread(buf, buf_size, 1, fp);
  return true;
}

bool File::ReadAllLines(const char *file_name,
                        std::vector<std::string> *all_lines) {
  ifstream iss(file_name);
  if (!iss.is_open() || all_lines == NULL) {
    return false;
  }
  all_lines->clear();
  std::string line;
  while (std::getline(iss, line)) {
    all_lines->push_back(line);
  }
  iss.close();
  return true;
}
}  // namespace idec

