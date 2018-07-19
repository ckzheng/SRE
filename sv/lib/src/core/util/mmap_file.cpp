#include "util/mmap_file.h"
#include "base/idec_types.h"
#include "base/log_message.h"
#include <iostream>
#include <assert.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <io.h>
#pragma warning (disable: 4996)
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace idec {

MemoryMapFile::MemoryMapFile() {
  file_flags_ =
#if defined(_WIN32) || defined(_WIN64)
    0 // MapOrThrow ignores flags on windows
#elif defined(MAP_FILE)
    MAP_FILE | MAP_SHARED
#else
    MAP_SHARED
#endif
    ;
  read_method_ = kLoadNone;
  fd_ = -1;;

  // the data ptr
  alloc_method_ = kNoneAllocated;
  data_ = NULL;
  size_ = 0;;
}

MemoryMapFile::~MemoryMapFile() {
  Close();
}

IDEC_RETCODE MemoryMapFile::Close() {
  if (NULL != data_ && alloc_method_ == kMmapAllocated) {
    SyncOrThrow(data_, size_);
    UnmapOrThrow(data_, size_);
  } else if (NULL != data_ && alloc_method_ == kMallocAllocated) {
    free(data_);
  }
  size_ = 0;
  data_ = NULL;
  alloc_method_ = kNoneAllocated;

  if (fd_ != -1) {
#if defined(_WIN32) || defined(_WIN64)
    _close(fd_);

#else
    close(fd_);

#endif
  }
  fd_ = -1;

  return IDEC_SUCCESS;
}

IDEC_RETCODE  MemoryMapFile::OpenForRead(ReadMethod method, const char *fname,
    uint64 offset) {
  IDEC_RETCODE ret = IDEC_SUCCESS;
  ret = OpenReadOrThrow(fname, &fd_);
  if (ret == IDEC_SUCCESS) {
    int64 size = SizeFile(fd_);
    MapRead(method, fd_, offset, size);
  }
  return ret;
}

IDEC_RETCODE MemoryMapFile::OpenForRead(ReadMethod method, const char *fname) {
  return OpenForRead(method, fname, 0);
}

IDEC_RETCODE MemoryMapFile::OpenForWrite(const char *fname, std::size_t size) {
  IDEC_RETCODE ret = CreateOrThrow(fname, &fd_);
  data_ = MapZeroedWrite(fd_, size);
  alloc_method_ = kMmapAllocated;
  return ret;
}

void *MemoryMapFile::MapZeroedWrite(int fd, std::size_t size) {
  ResizeOrThrow(fd, 0);
  ResizeOrThrow(fd, size);
  return MapOrThrow(size, true, file_flags_, false, fd, 0);
}

void *MemoryMapFile::MapOrThrow(std::size_t size, bool for_write, int flags,
                                bool prefault, int fd, uint64 offset) {
#ifdef MAP_POPULATE // Linux specific
  if (prefault) {
    flags |= MAP_POPULATE;
  }
#endif
#if defined(_WIN32) || defined(_WIN64)
  int protectC = for_write ? PAGE_READWRITE : PAGE_READONLY;
  int protectM = for_write ? FILE_MAP_WRITE : FILE_MAP_READ;
  uint64 total_size = size + offset;
  HANDLE hMapping = CreateFileMapping((HANDLE)_get_osfhandle(fd), NULL, protectC,
                                      total_size >> 32, static_cast<DWORD>(total_size), NULL);
  //UTIL_THROW_IF(!hMapping, ErrnoException, "CreateFileMapping failed");
  LPVOID ret = MapViewOfFile(hMapping, protectM, offset >> 32, (DWORD)offset,
                             size);
  CloseHandle(hMapping);
  //UTIL_THROW_IF(!ret, ErrnoException, "MapViewOfFile failed");
#else
  int protect = for_write ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void *ret = mmap(NULL, size, protect, flags, fd, offset);
  //UTIL_THROW_IF(ret == MAP_FAILED, ErrnoException, "mmap failed for size " << size << " at offset " << offset);
#endif
  return ret;
}

IDEC_RETCODE MemoryMapFile::MapRead(MemoryMapFile::ReadMethod method, int fd,
                                    uint64 offset, std::size_t size) {
  switch (method) {
  case kLoadlazy:
    data_ = MapOrThrow(size, false, file_flags_, false, fd, offset);
    size_ = size;
    alloc_method_ = kMmapAllocated;
    break;
  case kLoadPopulateOrLazy:
#ifdef MAP_POPULATE
  case kLoadPopulateOrMalloc:
#endif
    data_ = MapOrThrow(size, false, file_flags_, false, fd, offset);
    size_ = size;
    alloc_method_ = kMmapAllocated;
    break;
#ifndef MAP_POPULATE
  case kLoadPopulateOrMalloc:
#endif
  case kLoadMallocRead:

    data_ = malloc(size);
    if (data_ == NULL) {
      return IDEC_OUT_OF_MEMORY;
    }
    size_ = size;
    alloc_method_ = kMallocAllocated;
    SeekOrThrow(fd, offset);
    ReadOrThrow(fd, data_, size_);
    break;
  default:
    break;
  }
  return IDEC_SUCCESS;
}


void MemoryMapFile::SyncOrThrow(void *start, size_t length) {
#if defined(_WIN32) || defined(_WIN64)
  ::FlushViewOfFile(start, length);
#else
  msync(start, length, MS_SYNC);
#endif
}

void MemoryMapFile::UnmapOrThrow(void *start, size_t length) {
#if defined(_WIN32) || defined(_WIN64)
  UnmapViewOfFile(start);
#else
  munmap(start, length);
#endif
}

IDEC_RETCODE MemoryMapFile::CreateOrThrow(const char *name, int *fd) {
  int ret;
#if defined(_WIN32) || defined(_WIN64)
  ret = _open(name, _O_CREAT | _O_TRUNC | _O_RDWR | _O_BINARY,
              _S_IREAD | _S_IWRITE);
#else
  ret = open(name, O_CREAT | O_TRUNC | O_RDWR,
             S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
#endif
  *fd =  ret;
  if (ret == -1) {
    IDEC_WARNING << "creating file failure " << name;
    return IDEC_OPEN_ERROR;
  }
  return IDEC_SUCCESS;
}

IDEC_RETCODE MemoryMapFile::OpenReadOrThrow(const char *name, int *fd) {
  int ret;
#if defined(_WIN32) || defined(_WIN64)
  ret = _open(name, _O_BINARY | _O_RDONLY);
#else
  ret = open(name, O_RDONLY);
#endif
  if (ret == -1) {
    IDEC_WARNING << "open file failed for: " << name << "\n";
    return IDEC_OPEN_ERROR;
  }
  *fd=  ret;
  return IDEC_SUCCESS;
}

namespace {
void InternalSeek(int fd, int64 off, int whence) {
#if defined(_WIN32) || defined(_WIN64)
  //UTIL_THROW_IF((__int64)-1 == _lseeki64(fd, off, whence), ErrnoException, "Windows seek failed");
  _lseeki64(fd, off, whence);

#else
  lseek(fd, off, whence);;
#endif
}
} // namespace


void MemoryMapFile::SeekOrThrow(int fd, uint64 off) {
  InternalSeek(fd, off, SEEK_SET);
}


std::size_t MemoryMapFile::PartialRead(int fd, void *to, std::size_t amount) {
#if defined(_WIN32) || defined(_WIN64)
  amount = std::min(static_cast<std::size_t>(INT_MAX), amount);
  int ret = _read(fd, to, (uint32)amount);
#else
  ssize_t ret = read(fd, to, amount);
#endif
  //UTIL_THROW_IF(ret < 0, ErrnoException, "Reading " << amount << " from fd " << fd << " failed.");
  return static_cast<std::size_t>(ret);
}

void MemoryMapFile::ReadOrThrow(int fd, void *to_void, std::size_t amount) {
  uint8_t *to = static_cast<uint8_t *>(to_void);
  while (amount) {
    std::size_t ret = PartialRead(fd, to, amount);
    //UTIL_THROW_IF(ret == 0, EndOfFileException, "Hit EOF in fd " << fd << " but there should be " << amount << " more bytes to read.");
    amount -= ret;
    to += ret;
  }
}

uint64 MemoryMapFile::SizeFile(int fd) {
#if defined(_WIN32) || defined(_WIN64)
  __int64 ret = _filelengthi64(fd);
  return (ret == -1) ? kBadFileSize : ret;
#else
  struct stat sb;
  if (fstat(fd, &sb) == -1 || (!sb.st_size
                               && !S_ISREG(sb.st_mode))) return kBadFileSize;
  return sb.st_size;
#endif
}

void MemoryMapFile::ResizeOrThrow(int fd, uint64 to) {
#if defined(_WIN32) || defined(_WIN64)
  _chsize_s(fd, to);
#else
  ftruncate(fd, to);
#endif
}


}


