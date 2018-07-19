#include <cassert>
#include <algorithm>
#include "base/block_mem_pool.h"
#include "base/log_message.h"

namespace idec {
BlockMemPool::BlockMemPool(int elem_byte_len, int realloc_amount,
                           CachMethod cache_method) {

  if (sizeof(char) != 1) {
    IDEC_ERROR << "BlockMemPool::BlockMemPool - sizeof(char) assumption wrong";
  }
  cache_method_ = cache_method;
  realloc_amount_ = realloc_amount;
  if (cache_method == kCacheFreeList) {
    elem_byte_len_for_user_ = elem_byte_len;
    elem_byte_len_for_alloc_ = std::max(elem_byte_len, (int)sizeof(void *));
  } else {
    elem_byte_len_for_user_ = elem_byte_len;
    elem_byte_len_for_alloc_ = elem_byte_len;
  }
  realloc_byte_len_ = elem_byte_len_for_alloc_ * realloc_amount;

  num_total_ = 0;
  num_free_ = 0;
  num_used_ = 0;
  num_allocs_ = 0;
  get_cnt_ = 0;
  ret_cnt_ = 0;
  allocs_ = NULL;
  free_elems_arr_ = NULL;
}


BlockMemPool::~BlockMemPool() {
  PurgeMemory();
}

void BlockMemPool::PurgeMemory() {
  if (allocs_ != NULL) {
    for (int i = 0; i < num_allocs_; i++)
      delete[] allocs_[i];
    ::free(allocs_);
  }

  PurgeMemoryCacheArray();
  PurgeMemoryCacheList();
  num_total_ = 0;
  num_free_ = 0;
  num_used_ = 0;
  num_allocs_ = 0;
  get_cnt_ = 0;
  ret_cnt_ = 0;
  allocs_ = NULL;
}

void  BlockMemPool::PurgeMemoryCacheArray() {
  if (cache_method_ == kCacheArray) {
    if (free_elems_arr_ != NULL)
      ::free(free_elems_arr_);
  }
  free_elems_arr_ = NULL;
}

void *BlockMemPool::GetElem() {
  void *ptr = NULL;
  if (cache_method_ == kCacheArray) {
    ptr =  GetElemCacheArray();
  } else {
    ptr = GetElemCacheList();
  }
  num_free_--;
  num_used_++;
  get_cnt_++;
  return ptr;
}

void *BlockMemPool::GetElemCacheArray() {
  if (num_free_ == 0) {
    allocs_ = (char **)realloc(allocs_, (num_allocs_ + 1) * sizeof(char *));
    allocs_[num_allocs_] = new char[realloc_byte_len_];

    free_elems_arr_ = (char **)realloc(
                        free_elems_arr_, (num_total_ + realloc_amount_)*sizeof(char *)
                      );
    IDEC_ASSERT(free_elems_arr_ != NULL);
    for (int i = 0; i < realloc_amount_; i++)
      free_elems_arr_[i] = allocs_[num_allocs_] + elem_byte_len_for_alloc_ * i;

    num_total_ += realloc_amount_;
    num_free_ += realloc_amount_;
    num_allocs_++;
  }
  return free_elems_arr_[num_free_];
}

void *BlockMemPool::GetElemCacheList() {
  if (free_elems_list_.Empty()) {
    allocs_ = (char **)realloc(allocs_, (num_allocs_ + 1) * sizeof(char *));
    allocs_[num_allocs_] = new char[realloc_byte_len_];
    IDEC_ASSERT(allocs_[num_allocs_] != NULL);
    for (int i = 0; i < realloc_amount_; i++) {
      free_elems_list_.Push(allocs_[num_allocs_] + elem_byte_len_for_alloc_ * i);
    }
    num_total_ += realloc_amount_;
    num_free_ += realloc_amount_;
    num_allocs_++;
  }
  return free_elems_list_.Pop();
}


void BlockMemPool::ReturnElem(void *elem) {
  if (cache_method_ == kCacheArray) {
    ReturnElemCacheArray(elem);
  } else {
    ReturnElemCacheList(elem);
  }
  num_free_++;
  num_used_--;
  ret_cnt_++;
}

void BlockMemPool::ReturnElemCacheArray(void *elem) {
  free_elems_arr_[num_free_] = (char *)elem;
}

void BlockMemPool::ReturnElemCacheList(void *elem) {
  free_elems_list_.Push(elem);
}

}
