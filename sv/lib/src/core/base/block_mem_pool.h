#ifndef BLOCK_MEM_POOL
#define BLOCK_MEM_POOL

#include <cstdlib>
#include <cstring>

namespace idec {
namespace MemoryPool {
class FreeList {
 public:
  FreeList()
    : head_(0) {	// construct with empty list
  }
  bool Push(void *ptr) {	//push onto free list
    ((node *)ptr)->next = head_;
    head_ = (node *)ptr;
    return (true);
  }

  void *Pop() {	// pop node from free list
    void *ptr = head_;
    if (ptr != 0) {	// relink
      head_ = head_->next;
    }
    return (ptr);
  }

  bool Empty() {
    return head_ == 0;
  }
 private:
  struct node {// list node
    node *next;
  };
  node *head_;
};
}

class BlockMemPool {
 public:
  enum CachMethod {
    kCacheArray=0,
    kCacheFreeList=1
  };

 public:
  BlockMemPool(int elemByteLen, int reallocAmount,
               CachMethod cache_method = kCacheFreeList);
  virtual ~BlockMemPool();

  void *GetElem();
  void ReturnElem(void *elem);
  int GetCount() { return get_cnt_; }
  int NumUsed() { return num_used_; }

  bool IsAllFreed() {
    return (num_free_ == num_total_);
  }

  void *Malloc() { return GetElem(); }
  void *MallocZ() { void *p = Malloc(); memset(p, 0, elem_byte_len_for_user_); return p; }
  void Free(void *elem) { ReturnElem(elem); }
  void PurgeMemory();
 private:
  void *GetElemCacheArray();
  void *GetElemCacheList();
  void  ReturnElemCacheArray(void *elem);
  void  ReturnElemCacheList(void *elem);
  void  PurgeMemoryCacheArray();
  void  PurgeMemoryCacheList() {};

 private:
  int elem_byte_len_for_user_;   // size of element visible to user
  int elem_byte_len_for_alloc_;  // size of element for actually allocation
  int realloc_amount_;
  int realloc_byte_len_;
  int num_total_;
  int num_free_;
  int num_used_;
  int num_allocs_;
  int get_cnt_;
  int ret_cnt_;
  char **allocs_;


  // which cache method? array or free list
  CachMethod cache_method_;

  // method 0: cache_array
  char **free_elems_arr_;
  // method 1: free_list
  MemoryPool::FreeList free_elems_list_;
};


}

#endif


