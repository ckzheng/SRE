#ifndef UTIL_MMAP_FILE_H
#define UTIL_MMAP_FILE_H

#include <fstream>
#include <iostream>
#include "base/idec_return_code.h"
#include "base/idec_types.h"


namespace idec {
const uint64 kBadFileSize = (uint64)-1;

class MemoryMapFile {
 public:
  enum ReadMethod {
    // mmap with no prepopulate
    kLoadlazy,
    // On linux, pass MAP_POPULATE to mmap.
    kLoadPopulateOrLazy,
    // Populate on Linux.  malloc and read on non-Linux.
    kLoadPopulateOrMalloc,
    // malloc and read.
    kLoadMallocRead,
    kLoadNone
  };
  MemoryMapFile();
  ~MemoryMapFile();

  IDEC_RETCODE OpenForRead(ReadMethod method, const char *fname, uint64 offset);
  IDEC_RETCODE OpenForRead(ReadMethod method, const char *fname);
  IDEC_RETCODE OpenForWrite(const char *fname, std::size_t size);
  IDEC_RETCODE Close();
  void *GetPointer(void) const { return data_; };
  size_t GetSize(void) const { return size_; };


 private:
  typedef enum { kMmapAllocated, kMallocAllocated, kNoneAllocated } AllocMethod;
  int file_flags_;
  ReadMethod read_method_;
  int fd_;

  // the data ptrs
  AllocMethod alloc_method_;
  void *data_;
  size_t size_;

  // Wrapper around mmap to check it worked and hide some platform macros.
  void *MapOrThrow(std::size_t size, bool for_write, int flags, bool prefault,
                   int fd, uint64 offset = 0);
  IDEC_RETCODE MapRead(ReadMethod method, int fd, uint64 offset,
                       std::size_t size);
  void *MapZeroedWrite(int fd, std::size_t size);
  void SyncOrThrow(void *start, size_t length);
  void UnmapOrThrow(void *start, size_t length);

  // Wrapper around file describe operations
  IDEC_RETCODE OpenReadOrThrow(const char *name, int *fd);
  IDEC_RETCODE CreateOrThrow(const char *name, int *fd);
  void SeekOrThrow(int fd, uint64 off);
  std::size_t PartialRead(int fd, void *to, std::size_t amount);
  void ReadOrThrow(int fd, void *to_void, std::size_t amount);
  uint64 SizeFile(int fd);
  void ResizeOrThrow(int fd, uint64 to);
};
};    // end-of-namespace

#endif
